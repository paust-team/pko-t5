#!/bin/bash

set -e

curl -fsSL https://packages.redis.io/gpg | gpg -n --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list

apt-get update
apt-get install -y redis
service redis-server start