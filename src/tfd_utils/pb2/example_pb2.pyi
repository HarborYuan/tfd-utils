from tfd_utils.pb2 import feature_pb2 as _feature_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Example(_message.Message):
    __slots__ = ("features",)
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    features: _feature_pb2.Features
    def __init__(self, features: _Optional[_Union[_feature_pb2.Features, _Mapping]] = ...) -> None: ...

class SequenceExample(_message.Message):
    __slots__ = ("context", "feature_lists")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    FEATURE_LISTS_FIELD_NUMBER: _ClassVar[int]
    context: _feature_pb2.Features
    feature_lists: _feature_pb2.FeatureLists
    def __init__(self, context: _Optional[_Union[_feature_pb2.Features, _Mapping]] = ..., feature_lists: _Optional[_Union[_feature_pb2.FeatureLists, _Mapping]] = ...) -> None: ...
