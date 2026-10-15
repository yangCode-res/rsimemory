#!/usr/bin/env python3
"""Loopback TCP <-> private UNIX socket bridge for a single account's netns."""
import argparse
import asyncio
import os
from pathlib import Path


async def copy(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except (ConnectionError, OSError):
        pass


async def handle(reader, writer, socket_path, port, inner):
    other_writer = None
    try:
        if inner:
            other_reader, other_writer = await asyncio.open_connection('127.0.0.1', port)
        else:
            other_reader, other_writer = await asyncio.open_unix_connection(str(socket_path))
        await asyncio.gather(copy(reader, other_writer), copy(other_reader, writer))
    except (ConnectionError, OSError):
        pass
    finally:
        writer.close()
        if other_writer:
            other_writer.close()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--inner', action='store_true')
    args = parser.parse_args()
    servers = []
    for port in (3000, 9092):
        path = args.data / 'state' / ('bridge-' + str(port) + '.sock')
        async def callback(reader, writer, path=path, port=port):
            await handle(reader, writer, path, port, args.inner)
        if args.inner:
            path.unlink(missing_ok=True)
            server = await asyncio.start_unix_server(callback, path=str(path))
            os.chmod(path, 0o600)
        else:
            server = await asyncio.start_server(callback, host='127.0.0.1', port=port)
        servers.append(server)
    await asyncio.gather(*(server.serve_forever() for server in servers))


if __name__ == '__main__':
    asyncio.run(main())
