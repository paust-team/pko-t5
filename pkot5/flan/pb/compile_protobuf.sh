#!/bin/bash

set -e
cd $(dirname $0)

python3 -m grpc_tools.protoc -I./ --python_out=. --pyi_out=. --grpc_python_out=. ./flan.proto
sed 's/^import flan_pb2/from \. import flan_pb2/' ./flan_pb2_grpc.py > ./flan_pb2_grpc.py.1
mv ./flan_pb2_grpc.py.1 ./flan_pb2_grpc.py