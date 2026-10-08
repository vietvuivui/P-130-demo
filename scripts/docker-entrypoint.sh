#!/bin/sh
# Disk gắn ngoài (Render, docker volume) thường thuộc root: chuẩn bị quyền ghi rồi hạ xuống appuser để chạy server.
set -e
if [ "$(id -u)" = "0" ]; then
    mkdir -p /app/data
    chown -R appuser:appuser /app/data
    exec setpriv --reuid=appuser --regid=appuser --init-groups "$@"
fi
exec "$@"
