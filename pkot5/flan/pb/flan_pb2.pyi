from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GetReq(_message.Message):
    __slots__ = ["index"]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    index: int
    def __init__(self, index: _Optional[int] = ...) -> None: ...

class GetResp(_message.Message):
    __slots__ = ["input_ids", "target_ids"]
    INPUT_IDS_FIELD_NUMBER: _ClassVar[int]
    TARGET_IDS_FIELD_NUMBER: _ClassVar[int]
    input_ids: _containers.RepeatedScalarFieldContainer[int]
    target_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, input_ids: _Optional[_Iterable[int]] = ..., target_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class MetadataReq(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class MetadataResp(_message.Message):
    __slots__ = ["total"]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    total: int
    def __init__(self, total: _Optional[int] = ...) -> None: ...
