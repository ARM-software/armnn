# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
import inspect
import sys
import logging

from ._generated.pyarmnn_version import GetVersion, GetMajorVersion, GetMinorVersion

# Parsers

try:
    from ._generated.pyarmnn_onnxparser import IOnnxParser
except ImportError as err:
    logger = logging.getLogger(__name__)
    message = "Your ArmNN library instance does not support Onnx models parser functionality. "
    logger.warning("%s Skipped IOnnxParser import.", message)
    logger.debug(str(err))


    def IOnnxParser():
        """In case people try importing without having Arm NN built with this parser."""
        raise RuntimeError(message)

try:
    from ._generated.pyarmnn_tfliteparser import ITfLiteParser, TfLiteParserOptions
except ImportError as err:
    logger = logging.getLogger(__name__)
    message = "Your ArmNN library instance does not support TF lite models parser functionality. "
    logger.warning("%s Skipped ITfLiteParser import.", message)
    logger.debug(str(err))


    def ITfLiteParser():
        """In case people try importing without having Arm NN built with this parser."""
        raise RuntimeError(message)

try:
    from ._generated.pyarmnn_deserializer import IDeserializer
except ImportError as err:
    logger = logging.getLogger(__name__)
    message = "Your ArmNN library instance does not have ArmNN model (.armnn) parser functionality. "
    logger.warning("%s Skipped IDeserializer import.", message)
    logger.debug(str(err))

    def IDeserializer():
        """In case people try importing without having ArmNN built with this parser."""
        raise RuntimeError(message)

# Network
from ._generated.pyarmnn import Optimize, OptimizerOptions, IOptimizedNetwork, IInputSlot, \
    IOutputSlot, IConnectableLayer, INetwork

# Backend
from ._generated.pyarmnn import BackendId
from ._generated.pyarmnn import IDeviceSpec
from ._generated.pyarmnn import BackendOptions, BackendOption

# Tensors
from ._generated.pyarmnn import TensorInfo, TensorShape

# Runtime
from ._generated.pyarmnn import IRuntime, CreationOptions, INetworkProperties

# Profiler
from ._generated.pyarmnn import IProfiler

# Types
from ._generated.pyarmnn import DataType_Float16, DataType_Float32, DataType_QAsymmU8, DataType_Signed32, \
    DataType_Boolean, DataType_QSymmS16, DataType_QSymmS8, DataType_QAsymmS8, ShapeInferenceMethod_ValidateOnly, \
    ShapeInferenceMethod_InferAndValidate
from ._generated.pyarmnn import DataLayout_NCHW, DataLayout_NHWC, DataLayout_NCDHW, DataLayout_NDHWC
from ._generated.pyarmnn import MemorySource_Malloc, MemorySource_Undefined, MemorySource_DmaBuf, \
    MemorySource_DmaBufProtected
from ._generated.pyarmnn import ProfilingDetailsMethod_Undefined, ProfilingDetailsMethod_DetailsWithEvents, \
    ProfilingDetailsMethod_DetailsOnly

from ._generated.pyarmnn import ActivationFunction_Abs, ActivationFunction_BoundedReLu, ActivationFunction_LeakyReLu, \
    ActivationFunction_Linear, ActivationFunction_ReLu, ActivationFunction_Sigmoid, ActivationFunction_SoftReLu, \
    ActivationFunction_Sqrt, ActivationFunction_Square, ActivationFunction_TanH, ActivationDescriptor
from ._generated.pyarmnn import ArgMinMaxFunction_Max, ArgMinMaxFunction_Min, ArgMinMaxDescriptor
from ._generated.pyarmnn import BatchNormalizationDescriptor, BatchToSpaceNdDescriptor
from ._generated.pyarmnn import ChannelShuffleDescriptor, ComparisonDescriptor, ComparisonOperation_Equal, \
    ComparisonOperation_Greater, ComparisonOperation_GreaterOrEqual, ComparisonOperation_Less, \
    ComparisonOperation_LessOrEqual, ComparisonOperation_NotEqual
from ._generated.pyarmnn import UnaryOperation_Abs, UnaryOperation_Exp, UnaryOperation_Sqrt, UnaryOperation_Rsqrt, \
    UnaryOperation_Neg, ElementwiseUnaryDescriptor
from ._generated.pyarmnn import LogicalBinaryOperation_LogicalAnd, LogicalBinaryOperation_LogicalOr, \
    LogicalBinaryDescriptor
from ._generated.pyarmnn import Convolution2dDescriptor, Convolution3dDescriptor, DepthToSpaceDescriptor, \
    DepthwiseConvolution2dDescriptor, DetectionPostProcessDescriptor, FakeQuantizationDescriptor, FillDescriptor, \
    FullyConnectedDescriptor, GatherDescriptor, InstanceNormalizationDescriptor, LstmDescriptor, \
    L2NormalizationDescriptor, MeanDescriptor
from ._generated.pyarmnn import NormalizationAlgorithmChannel_Across, NormalizationAlgorithmChannel_Within, \
    NormalizationAlgorithmMethod_LocalBrightness, NormalizationAlgorithmMethod_LocalContrast, NormalizationDescriptor
from ._generated.pyarmnn import PaddingMode_Constant, PaddingMode_Reflect, PaddingMode_Symmetric, PadDescriptor
from ._generated.pyarmnn import PermutationVector, PermuteDescriptor
from ._generated.pyarmnn import OutputShapeRounding_Ceiling, OutputShapeRounding_Floor, \
    PaddingMethod_Exclude, PaddingMethod_IgnoreValue, PoolingAlgorithm_Average, PoolingAlgorithm_L2, \
    PoolingAlgorithm_Max, Pooling2dDescriptor, Pooling3dDescriptor
from ._generated.pyarmnn import ReduceDescriptor, ReduceOperation_Prod, ReduceOperation_Max, ReduceOperation_Mean, \
    ReduceOperation_Min, ReduceOperation_Sum
from ._generated.pyarmnn import ResizeMethod_Bilinear, ResizeMethod_NearestNeighbor, ResizeDescriptor, \
    ReshapeDescriptor, SliceDescriptor, SpaceToBatchNdDescriptor, SpaceToDepthDescriptor, StandInDescriptor, \
    StackDescriptor, StridedSliceDescriptor, SoftmaxDescriptor, TransposeConvolution2dDescriptor, \
    TransposeDescriptor, SplitterDescriptor
from ._generated.pyarmnn import ConcatDescriptor, CreateDescriptorForConcatenation

from ._generated.pyarmnn import LstmInputParams, QuantizedLstmInputParams

# Public API
# Quantization
from ._quantization.quantize_and_dequantize import quantize, dequantize

# Tensor
from ._tensor.tensor import Tensor
from ._tensor.const_tensor import ConstTensor
from ._tensor.workload_tensors import make_input_tensors, make_output_tensors, workload_tensors_to_ndarray

# Utilities
from ._utilities.profiling_helper import ProfilerData, get_profiling_data

from ._version import __version__, __arm_ml_version__

ARMNN_VERSION = GetVersion()


def __check_version():
    from ._version import check_armnn_version
    check_armnn_version(ARMNN_VERSION)


__check_version()

__all__ = []

__private_api_names = ['__check_version']

for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) or inspect.isfunction(obj):
        if name not in __private_api_names:
            __all__.append(name)
