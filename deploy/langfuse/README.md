# 超算 Langfuse 部署

2026-10-15 已在 `chaosuan` 的 `choud-mgt-09` 部署 Langfuse **v4.59.0 OSS**。网页、worker、PostgreSQL、ClickHouse、Redis、MinIO 六个服务均已启动；本机 SSH 隧道访问网页健康接口和对象存储健康接口均为 200。

## 实际运行方式

当前账号没有 Docker 管理员权限，也没有 subordinate UID/GID 段。使用已校验的静态 Podman 5.8.4、当前账号自己的 user/mount/network namespace 和单 UID/GID 映射启动，不需要 root。所有容器共享账号自己的隔离网络，在其中监听回环地址。`bridge.py` 通过私有 UNIX socket，仅将网页 3000 与对象存储 9092 转发至宿主 `127.0.0.1`。本机再通过 SSH 访问，不公开数据库端口。

镜像已按 `image-lock.json` 中的官方 amd64 摘要取得；服务软件归档保存为 `images/*.rootfs.tar`，镜像配置及归档 SHA256 缓存于 `config/images/` 并绑定镜像 ID。迁移后无需读取旧镜像仓库；启动时校验归档，在 `/tmp/rsi-langfuse-1035/rootfs/` 解压为可重建的软件缓存。宿主账号始终是 `juanliu`，不授予其他宿主 UID 的权限。

持久数据库、服务软件归档、静态运行时、日志和密钥统一放在共享持久存储：

```text
/file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi/observability/langfuse/
  .env                # 0600，初始化账号、项目 API keys、数据库密钥
  runtime/            # 已安装的静态 Podman bundle
  graphroot/          # 新容器引擎的元数据；不依赖旧镜像中间层
  images/             # 六个服务的私有软件归档，SHA256 绑定镜像 ID
  data/               # 数据库和对象存储
  config/             # 私有服务 env 文件、Redis 配置
  state/              # 保留状态目录；活动进程状态在 /tmp
  logs/               # 每个服务的日志
```

namespace/bridge 的 PID、UNIX socket、Podman runroot 和软件解压缓存放在权限 0700 的 `/tmp/rsi-langfuse-1035/`，可在重启后重建；此目录不保存数据库或用户轨迹。服务的软件原档和全部持久数据均在 `/file_storage01`。

启动器将新启动进程的 CPU affinity 限定到四个可用 CPU；ClickHouse 服务器内存上限 12 GiB，查询四线程、单查询内存 2 GiB；Node heap 上限每进程 2 GiB。未启用宿主 cgroups 管理，这些设置不构成整个栈的总 RSS 硬限制。遥测已关闭；隔离网络不配置外部出口，在线版本检查可能记录 DNS 失败，不影响轨迹入库。

## 启停和状态

在超算项目根目录运行：

```sh
cd /file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi
python3 deploy/langfuse/hpc.py status
python3 deploy/langfuse/hpc.py up
# 可指定单个服务，例如 web、worker、clickhouse
python3 deploy/langfuse/hpc.py stop web
python3 deploy/langfuse/hpc.py up web
# 停止六个容器，保留数据与 namespace/bridge 辅助进程
python3 deploy/langfuse/hpc.py stop
```

静态 runtime 和服务软件归档已安装在上述持久目录。`up` 可复用它们，并在 namespace 进程失效时重新创建；当前没有安装系统级开机自启。宿主重启后需手动执行 `up` 并检查健康接口。`compose.yaml` 是有常规 Docker/Compose 环境时的配置参考，当前超算实例使用 `hpc.py`，不执行 `init.sh up`。

2026-10-15 已停止旧服务后迁移数据库与对象存储，10,820 个文件共 465,171,072 字节的 SHA256 与源文件全部一致。新实例的六个服务均已启动，持久数据挂载全部指向上述 `/file_storage01` 目录；历史运行的 4,615 个 observation、4,608 个问题记录逐 ID 复核一致。旧存储上的服务与辅助进程已停用，旧副本暂留备份，不再作为运行目录。迁移核对回执保存在项目 `runs/langfuse-storage-migration-20261015.json`，不包含轨迹正文。

## 在电脑上访问

已建立本机 3000/9092 的 SSH 隧道。如隧道断开，在电脑终端重新执行并保持运行：

```sh
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:3000:127.0.0.1:3000 \
  -L 127.0.0.1:9092:127.0.0.1:9092 chaosuan
```

打开 <http://localhost:3000>。已初始化组织 `rsi-memory`、项目 `memory-policy-rsi`，账号为 `rsi-admin@rsi.local`；密码在服务器私有 `.env` 的 `LANGFUSE_INIT_USER_PASSWORD`，也已单独保存到电脑项目的 `runs/langfuse-access/login.txt`。API keys 留在服务器，不加入版本库。

此次创建的后台 SSH control socket 是 `/private/tmp/rsi-langfuse-ssh-control`；停止该隧道可执行 `ssh -S /private/tmp/rsi-langfuse-ssh-control -O exit chaosuan`，不会停止服务器容器。

## 已验证的轨迹

历史运行 146133 的导入目录为 `runs/langfuse-backfill-146133-v2/`。导入回执接受 4615 个 observation；进一步通过 **v4 Observations API v2** 分页读取，对照原始 ID，确认 4615 个唯一 observation、4609 条 trace、4608 个问题记录完全匹配，缺失和多余均为零。核对结果在该目录的 `verification.json`。另外导入了明确标记 fixture 的 240 个合成 observation 验证新进化闭环；不能将它们当成真实实验成绩。

历史决策树与反思：

- <http://localhost:3000/project/memory-policy-rsi/traces/ac9f73e55556abd6462be8f4e27bebea>
- <http://localhost:3000/project/memory-policy-rsi/traces/ac9f73e55556abd6462be8f4e27bebea?observation=5b0a9aaa3b2b2b6a>

树中包括 `diagnostic_bundle`、`reflection`、三个 `codegen` 与 `validation_and_cost_summary`。候选区分信息在 observation metadata 的 `policy`。历史时间轴是导入时间，不代表原始运行耗时；旧提案没有明确案例归因的部分标记未记录。详见 [轨迹说明](../../docs/LANGFUSE.md)。

## 实验自动登录入口

`hpc.py up` 同时启动 `auto_login.py`，只监听服务器 `127.0.0.1:3001`。
在电脑增加 SSH 转发 `-L 127.0.0.1:3001:127.0.0.1:3001`，打开
<http://localhost:3001/>，即可使用默认实验账号登录并跳转至 3000 的项目页。
仍需原有 3000/9092 转发。凭据只在服务器读取，浏览器接收 HttpOnly 会话 cookie。
这是共享实验账号入口：能访问该端口的人即获得默认账号权限，只通过既有 SSH 隧道使用。
无需修改 Langfuse 镜像、数据库或原有密码认证；停止辅助进程即可撤销自动登录入口。

2026-10-21：修复首页查询被全局内存上限中止的问题，将 ClickHouse 上限由 4 GiB 调为 12 GiB，mark cache 由 512 MiB 调为 128 MiB。保留单查询 2 GiB、四线程限制；已核验配置生效与项目聚合查询恢复。此限额不等于容器 RSS 硬上限。
