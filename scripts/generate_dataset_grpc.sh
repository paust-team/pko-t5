#!/bin/bash

cd pkot5/data
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./dataset.proto