#!/usr/bin/env bash
set -euo pipefail
mount --make-rprivate /
mount -t tmpfs -o size=64m tmpfs /run
mkdir -p /run/crun /run/libpod /run/user/0
chmod 700 /run/user/0
ip link set lo up
exec sleep infinity
