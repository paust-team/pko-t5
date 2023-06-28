# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import flan_pb2 as flan__pb2


class FlanProcessingDatasetStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Get = channel.unary_unary(
                '/flan.FlanProcessingDataset/Get',
                request_serializer=flan__pb2.GetReq.SerializeToString,
                response_deserializer=flan__pb2.GetResp.FromString,
                )
        self.Metadata = channel.unary_unary(
                '/flan.FlanProcessingDataset/Metadata',
                request_serializer=flan__pb2.MetadataReq.SerializeToString,
                response_deserializer=flan__pb2.MetadataResp.FromString,
                )


class FlanProcessingDatasetServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Get(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Metadata(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FlanProcessingDatasetServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Get': grpc.unary_unary_rpc_method_handler(
                    servicer.Get,
                    request_deserializer=flan__pb2.GetReq.FromString,
                    response_serializer=flan__pb2.GetResp.SerializeToString,
            ),
            'Metadata': grpc.unary_unary_rpc_method_handler(
                    servicer.Metadata,
                    request_deserializer=flan__pb2.MetadataReq.FromString,
                    response_serializer=flan__pb2.MetadataResp.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'flan.FlanProcessingDataset', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class FlanProcessingDataset(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Get(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/flan.FlanProcessingDataset/Get',
            flan__pb2.GetReq.SerializeToString,
            flan__pb2.GetResp.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Metadata(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/flan.FlanProcessingDataset/Metadata',
            flan__pb2.MetadataReq.SerializeToString,
            flan__pb2.MetadataResp.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
