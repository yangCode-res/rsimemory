#!/usr/bin/env python3
"""Manage this account's loopback-only Langfuse stack in its own user namespace."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import stat
import sys
import time
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get('LANGFUSE_DATA_ROOT', str(HERE.parent.parent / 'observability/langfuse')))
RUNTIME = Path(os.environ.get('LANGFUSE_RUNTIME_DIR', '/tmp/rsi-langfuse-' + str(os.getuid())))
PODMAN = DATA / 'runtime/usr/local/bin/podman'
PREFIX = 'rsi-langfuse-'
SERVICES = ('postgres', 'redis', 'minio', 'clickhouse', 'web', 'worker')
IMAGES = {
    'postgres': '327daa8fae7178d61f93142f146b098467b345e10997b9eb79f63bd58e5c8f3c',
    'redis': 'd24fb73aa9caa76060b48f711af3831f1e95286c745da622c00419ed864cd291',
    'minio': '78a4e522310bd8d7d3ab547e56774a0a85cfba0b14ed5a0f2242781112c9a787',
    'clickhouse': '0e76d9807edfac0345554816cd7f54aa9b650892848809932557c46e46bc3450',
    'web': '744d35c73b5dc695956c7f2c09a18f247b50101e3b0fedc8f4bacbff32c6ee6c',
    'worker': 'efb119d670284c04b93b9ec746cfc87a548797bb42d9d624e0842ef55a407c41',
}


def namespace_prefix():
    pid = (RUNTIME / 'state/podman-ns.pid').read_text().strip()
    cpus = ','.join(map(str, sorted(os.sched_getaffinity(0))[:4]))
    return ['taskset', '-c', cpus, 'nsenter', '--target', pid, '--user', '--mount',
            '--net', '--preserve-credentials']


def engine(*args, check=True):
    command = namespace_prefix() + [
        'env', 'CONTAINERS_CONF=' + str(DATA / 'config/containers/containers.conf'),
        'XDG_RUNTIME_DIR=' + str(RUNTIME / 'run'),
        'CONTAINERS_STORAGE_CONF=' + str(DATA / 'config/containers/storage.conf'),
        'PATH=' + str(DATA / 'runtime/usr/local/bin') + ':/usr/bin:/bin', str(PODMAN), *args]
    return subprocess.run(command, check=check, capture_output=True, text=True)


def process_alive(pid_file, expected):
    try:
        pid = int(pid_file.read_text())
        command = Path('/proc/' + str(pid) + '/cmdline').read_bytes()
        return os.stat('/proc/' + str(pid)).st_uid == os.getuid() and expected.encode() in command
    except (OSError, ValueError):
        return False


def ensure_namespace():
    if not PODMAN.exists():
        raise RuntimeError('Pinned Podman bundle is missing from ' + str(PODMAN.parent))
    RUNTIME.mkdir(mode=0o700, exist_ok=True)
    if RUNTIME.is_symlink() or RUNTIME.stat().st_uid != os.getuid():
        raise RuntimeError('Runtime directory must belong to this account')
    RUNTIME.chmod(0o700)
    for name in ('state','run'):
        (RUNTIME / name).mkdir(mode=0o700, exist_ok=True)
    pid_file = RUNTIME / 'state/podman-ns.pid'
    if process_alive(pid_file, 'namespace.sh'):
        return
    cpus = ','.join(map(str, sorted(os.sched_getaffinity(0))[:4]))
    with open(DATA / 'logs/namespace.log', 'ab') as log:
        process = subprocess.Popen(['taskset', '-c', cpus, 'unshare', '--user',
            '--map-root-user', '--mount', '--net', '--fork', 'bash', str(HERE / 'namespace.sh')],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    private_file(pid_file, str(process.pid))
    time.sleep(.5)
    if process.poll() is not None:
        raise RuntimeError('Namespace startup failed; inspect private namespace.log')


def ensure_bridges():
    for inner in (True, False):
        name = 'bridge-inner' if inner else 'bridge-outer'
        pid_file = RUNTIME / 'state' / (name + '.pid')
        if process_alive(pid_file, str(HERE / 'bridge.py')):
            continue
        command = [sys.executable, str(HERE / 'bridge.py'), '--data', str(RUNTIME)]
        if inner:
            command = namespace_prefix() + command + ['--inner']
        with open(DATA / 'logs' / (name + '.log'), 'ab') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=log, start_new_session=True)
        private_file(pid_file, str(process.pid))
        time.sleep(.3)
        if process.poll() is not None:
            raise RuntimeError(name + ' startup failed; inspect its private log')


def ensure_auto_login():
    pid_file = RUNTIME / 'state/auto-login.pid'
    script = HERE / 'auto_login.py'
    if process_alive(pid_file, str(script)):
        return
    with open(DATA / 'logs/auto-login.log', 'ab') as log:
        process = subprocess.Popen([sys.executable, str(script)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    private_file(pid_file, str(process.pid))
    time.sleep(.3)
    if process.poll() is not None:
        raise RuntimeError('Auto-login helper startup failed')


def private_file(path, text):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, text.encode())
    finally:
        os.close(fd)


def environment():
    return dict(line.split('=', 1) for line in (DATA / '.env').read_text().splitlines()
                if line and not line.startswith('#') and '=' in line)


def app_environment(env):
    values = {key: value for key, value in env.items()
              if key.startswith('LANGFUSE_INIT_') or key in
              ('SALT', 'ENCRYPTION_KEY', 'NEXTAUTH_SECRET', 'NEXTAUTH_URL', 'REDIS_AUTH')}
    values.update({
        'DATABASE_URL': 'postgresql://postgres:' + env['POSTGRES_PASSWORD'] + '@127.0.0.1:5432/postgres',
        'TELEMETRY_ENABLED': 'false', 'NEXT_TELEMETRY_DISABLED': '1',
        'CLICKHOUSE_MIGRATION_URL': 'clickhouse://127.0.0.1:9000',
        'CLICKHOUSE_URL': 'http://127.0.0.1:8123', 'CLICKHOUSE_USER': 'clickhouse',
        'CLICKHOUSE_PASSWORD': env['CLICKHOUSE_PASSWORD'], 'CLICKHOUSE_CLUSTER_ENABLED': 'false',
        'REDIS_HOST': '127.0.0.1', 'REDIS_PORT': '6379', 'REDIS_TLS_ENABLED': 'false',
        'HOSTNAME': '127.0.0.1', 'HOST': '127.0.0.1',
        'NODE_OPTIONS': '--max-old-space-size=2048 --require=/run/rsi-loopback.cjs',
    })
    for purpose in ('EVENT', 'MEDIA'):
        base = 'LANGFUSE_S3_' + purpose + '_UPLOAD_'
        values.update({base + 'BUCKET': 'langfuse', base + 'REGION': 'us-east-1',
            base + 'ACCESS_KEY_ID': env['MINIO_ROOT_USER'],
            base + 'SECRET_ACCESS_KEY': env['MINIO_ROOT_PASSWORD'],
            base + 'ENDPOINT': 'http://localhost:9092', base + 'FORCE_PATH_STYLE': 'true',
            base + 'PREFIX': 'events/' if purpose == 'EVENT' else 'media/'})
    values['LANGFUSE_S3_MEDIA_UPLOAD_INTERNAL_ENDPOINT'] = 'http://127.0.0.1:9092'
    return values


def running(name):
    result = engine('container', 'inspect', PREFIX + name, '--format', '{{.State.Running}}', check=False)
    return result.returncode == 0 and result.stdout.strip() == 'true'


def launch(name):
    if running(name):
        print(name + ': already running', flush=True)
        return
    # Only remove this launcher's stopped container; never touch another stack.
    engine('rm', PREFIX + name, check=False)
    archive = DATA / 'images' / (name + '.rootfs.tar')
    rootfs = (RUNTIME if archive.exists() else DATA) / 'rootfs' / name
    copy_marker = (RUNTIME if archive.exists() else DATA) / 'state' / ('rootfs-' + name + '.ready')
    metadata_file = DATA / 'config/images' / (name + '.json')
    ready = copy_marker.exists() if archive.exists() else (
        (rootfs / '.rsi-copy-complete').exists() or copy_marker.exists())
    if (ready or archive.exists()) and metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        if metadata.get('image_id') != IMAGES[name]:
            raise RuntimeError('Cached rootfs image identity mismatch for ' + name)
        config = metadata['config']
    else:
        config = json.loads(engine('image', 'inspect', IMAGES[name]).stdout)[0]['Config']
    if archive.exists() and ready:
        ready = copy_marker.read_text()==metadata.get('rootfs_sha256')
    if archive.exists() and not ready:
        expected = metadata.get('rootfs_sha256')
        digest = hashlib.sha256()
        with archive.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                digest.update(block)
        if not expected or digest.hexdigest()!=expected:
            raise RuntimeError('Rootfs archive checksum mismatch for ' + name)
        if rootfs.exists():
            for directory,_,_ in os.walk(rootfs):
                path=Path(directory);path.chmod(stat.S_IMODE(path.stat().st_mode)|stat.S_IRWXU)
            shutil.rmtree(rootfs)
        rootfs.mkdir(parents=True,mode=0o700)
        subprocess.run(['tar','--extract','--no-same-owner','--file',str(archive),
                        '--directory',str(rootfs)],check=True)
        private_file(copy_marker,expected)
        ready=True
    if not ready:
        source = Path(engine('image', 'mount', IMAGES[name]).stdout.strip())
        if rootfs.exists():
            # Resume only this launcher's incomplete image copy; source images stay intact.
            for directory, _, _ in os.walk(rootfs):
                path = Path(directory)
                path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IRWXU)
            shutil.rmtree(rootfs)
        rootfs.mkdir(parents=True, mode=0o700, exist_ok=True)
        unreadable = []
        try:
            for directory, _, files in os.walk(source):
                for filename in files:
                    file = Path(directory) / filename
                    mode = file.lstat().st_mode
                    if stat.S_ISREG(mode) and not mode & stat.S_IRUSR:
                        unreadable.append((file, stat.S_IMODE(mode)))
                        file.chmod(stat.S_IMODE(mode) | stat.S_IRUSR)
            subprocess.run(['cp', '-a', '--reflink=auto', str(source) + '/.', str(rootfs)], check=True)
        finally:
            for file, mode in unreadable:
                file.chmod(mode)
                target = rootfs / file.relative_to(source)
                if target.exists():
                    target.chmod(mode)
        private_file(copy_marker, IMAGES[name])
    if not archive.exists():
        private_file(metadata_file, json.dumps({'image_id':IMAGES[name], 'config':config}))
    uid = {'postgres': 999, 'redis': 999, 'minio': 65532, 'clickhouse': 101,
           'web': 1001, 'worker': 1001}[name]
    env = environment()
    values = dict(item.split('=', 1) for item in config.get('Env') or [] if '=' in item)
    values['HOME'] = '/tmp'
    command = (config.get('Entrypoint') or []) + (config.get('Cmd') or [])
    mounts = []
    if name == 'postgres':
        values.update(POSTGRES_USER='postgres', POSTGRES_PASSWORD=env['POSTGRES_PASSWORD'],
                      POSTGRES_DB='postgres', TZ='UTC', PGTZ='UTC')
        mounts.append((DATA / 'data/postgres', '/var/lib/postgresql/data', False))
        command += ['-c', 'listen_addresses=127.0.0.1']
    elif name == 'redis':
        private_file(DATA / 'config/redis.conf', 'bind 127.0.0.1\nport 6379\nprotected-mode yes\n'
            'dir /data\nappendonly yes\nmaxmemory 1073741824\nmaxmemory-policy noeviction\nrequirepass '
            + env['REDIS_AUTH'] + '\n')
        mounts.extend([(DATA / 'data/redis', '/data', False),
                       (DATA / 'config/redis.conf', '/run/redis.conf', True)])
        command = ['docker-entrypoint.sh', 'redis-server', '/run/redis.conf']
    elif name == 'minio':
        values.update({key: env[key] for key in ('MINIO_ROOT_USER', 'MINIO_ROOT_PASSWORD')})
        (DATA / 'data/minio/langfuse').mkdir(exist_ok=True)
        mounts.append((DATA / 'data/minio', '/data', False))
        command = ['/usr/bin/minio', 'server', '--address', '127.0.0.1:9092',
                   '--console-address', '127.0.0.1:9091', '/data']
    elif name == 'clickhouse':
        values.update(CLICKHOUSE_DB='default', CLICKHOUSE_USER='clickhouse',
                      CLICKHOUSE_PASSWORD=env['CLICKHOUSE_PASSWORD'], CLICKHOUSE_SKIP_USER_SETUP='0')
        mounts.extend([(DATA / 'data/clickhouse', '/var/lib/clickhouse', False),
            (DATA / 'data/clickhouse-logs', '/var/log/clickhouse-server', False),
            (HERE / 'clickhouse-limits.xml', '/etc/clickhouse-server/config.d/rsi-limits.xml', True),
            (HERE / 'clickhouse-users.xml', '/etc/clickhouse-server/users.d/rsi-limits.xml', True)])
    else:
        values.update(app_environment(env))
        values['PORT'] = '3000' if name == 'web' else '3030'
        # The worker's health server defaults to all interfaces. Restrict Node servers
        # at listen() without affecting outbound sockets or changing Langfuse code.
        private_file(DATA / 'config/loopback.cjs', '''const net=require('node:net');
const original=net.Server.prototype.listen;
net.Server.prototype.listen=function(...a){
 if(a[0] && typeof a[0]==='object' && !a[0].path) a[0]={...a[0],host:'127.0.0.1'};
 else if(typeof a[0]==='number' || (typeof a[0]==='string' && /^\\d+$/.test(a[0]))){
  if(typeof a[1]==='string') a[1]='127.0.0.1'; else a.splice(1,0,'127.0.0.1');
 }
 return original.apply(this,a);
};
''')
        mounts.append((DATA / 'config/loopback.cjs', '/run/rsi-loopback.cjs', True))
    env_file = DATA / 'config' / (name + '.env')
    private_file(env_file, ''.join(key + '=' + value + '\n' for key, value in values.items()))
    args = ['run', '--detach', '--name', PREFIX + name, '--rootfs',
        '--uidmap=' + str(uid) + ':0:1', '--gidmap=' + str(uid) + ':0:1',
        '--user=' + str(uid) + ':' + str(uid), '--network=host', '--cgroups=disabled',
        '--cap-drop=ALL', '--security-opt=no-new-privileges',
        '--mount=type=devpts,destination=/dev/pts,uid=' + str(uid) + ',gid=' + str(uid) + ',mode=0620',
        '--tmpfs=/tmp:rw,mode=1777',
        '--env-file', str(env_file), '--log-driver=k8s-file',
        '--log-opt=path=' + str(DATA / 'logs' / (name + '.log')), '--log-opt=max-size=10mb']
    if config.get('WorkingDir'):
        args.extend(['--workdir', config['WorkingDir']])
    for host, target, readonly in mounts:
        args.extend(['--volume', str(host) + ':' + target + (':ro' if readonly else ':rw')])
    result = engine(*args, str(rootfs), *command, check=False)
    if result.returncode:
        # Podman arguments contain only env-file paths, never secret values.
        raise RuntimeError(name + ': ' + result.stderr.strip())
    print(name + ': started', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('up', 'status', 'stop'))
    parser.add_argument('services', nargs='*')
    args = parser.parse_args()
    if any(name not in SERVICES for name in args.services):
        parser.error('services must be one of: ' + ', '.join(SERVICES))
    names = args.services or SERVICES
    if args.action == 'up':
        ensure_namespace()
        for name in names:
            launch(name)
        ensure_bridges()
        ensure_auto_login()
    elif args.action == 'stop':
        for name in reversed(names):
            engine('stop', '--time=30', PREFIX + name, check=False)
            print(name + ': stopped')
    else:
        for name in names:
            print(name + ': ' + ('running' if running(name) else 'stopped'))


if __name__ == '__main__':
    main()
