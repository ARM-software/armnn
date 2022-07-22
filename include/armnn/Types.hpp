//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <array>
#include <functional>
#include <stdint.h>
#include <chrono>
#include "BackendId.hpp"
#include "Exceptions.hpp"
#include "Deprecated.hpp"

namespace arm
{
namespace pipe
{

class ProfilingGuid;

} // namespace arm
} // namespace pipe

/// Define LayerGuid type.
using LayerGuid = arm::pipe::ProfilingGuid;

namespace armnn
{

constexpr unsigned int MaxNumOfTensorDimensions = 5U;

/// The lowest performance data capture interval we support is 10 miliseconds.
constexpr unsigned int LOWEST_CAPTURE_PERIOD = 10000u;

/// Variable to control expire rate of priority queue
constexpr unsigned int EXPIRE_RATE = 3U;

/// @enum Status enumeration
/// @var Status::Successful
/// @var Status::Failure
enum class Status
{
    Success = 0,
    Failure = 1
};

enum class DataType
{
    Float16  = 0,
    Float32  = 1,
    QAsymmU8 = 2,
    Signed32 = 3,
    Boolean  = 4,
    QSymmS16 = 5,
    QSymmS8  = 6,
    QAsymmS8 = 7,
    BFloat16 = 8,
    Signed64 = 9,
};

enum class DataLayout
{
    NCHW = 1,
    NHWC = 2,
    NDHWC = 3,
    NCDHW = 4
};

/// Define the behaviour of the internal profiler when outputting network details
enum class ProfilingDetailsMethod
{
    Undefined = 0,
    DetailsWithEvents = 1,
    DetailsOnly = 2
};


enum class QosExecPriority
{
    Low    = 0,
    Medium = 1,
    High   = 2
};

enum class ActivationFunction
{
    Sigmoid     = 0,
    TanH        = 1,
    Linear      = 2,
    ReLu        = 3,
    BoundedReLu = 4, ///< min(a, max(b, input)) ReLu1 & ReLu6.
    SoftReLu    = 5,
    LeakyReLu   = 6,
    Abs         = 7,
    Sqrt        = 8,
    Square      = 9,
    Elu         = 10,
    HardSwish   = 11
};

enum class ArgMinMaxFunction
{
    Min = 0,
    Max = 1
};

enum class ComparisonOperation
{
    Equal          = 0,
    Greater        = 1,
    GreaterOrEqual = 2,
    Less           = 3,
    LessOrEqual    = 4,
    NotEqual       = 5
};

enum class LogicalBinaryOperation
{
    LogicalAnd = 0,
    LogicalOr  = 1
};

enum class UnaryOperation
{
    Abs        = 0,
    Exp        = 1,
    Sqrt       = 2,
    Rsqrt      = 3,
    Neg        = 4,
    LogicalNot = 5,
    Log        = 6,
    Sin        = 7
};

enum class PoolingAlgorithm
{
    Max     = 0,
    Average = 1,
    L2      = 2
};

enum class ReduceOperation
{
    Sum  = 0,
    Max  = 1,
    Mean = 2,
    Min  = 3,
    Prod = 4
};

enum class ResizeMethod
{
    Bilinear        = 0,
    NearestNeighbor = 1
};

enum class Dimensionality
{
    NotSpecified = 0,
    Specified    = 1,
    Scalar       = 2
};

///
/// The padding method modifies the output of pooling layers.
/// In both supported methods, the values are ignored (they are
/// not even zeroes, which would make a difference for max pooling
/// a tensor with negative values). The difference between
/// IgnoreValue and Exclude is that the former counts the padding
/// fields in the divisor of Average and L2 pooling, while
/// Exclude does not.
///
enum class PaddingMethod
{
    /// The padding fields count, but are ignored
    IgnoreValue = 0,
    /// The padding fields don't count and are ignored
    Exclude     = 1
};

///
/// The padding mode controls whether the padding should be filled with constant values (Constant), or
/// reflect the input, either including the border values (Symmetric) or not (Reflect).
///
enum class PaddingMode
{
    Constant  = 0,
    Reflect   = 1,
    Symmetric = 2
};

enum class NormalizationAlgorithmChannel
{
    Across = 0,
    Within = 1
};

enum class NormalizationAlgorithmMethod
{
    /// Krichevsky 2012: Local Brightness Normalization
    LocalBrightness = 0,
    /// Jarret 2009: Local Contrast Normalization
    LocalContrast = 1
};

enum class OutputShapeRounding
{
    Floor       = 0,
    Ceiling     = 1
};

///
/// The ShapeInferenceMethod modify how the output shapes are treated.
/// When ValidateOnly is selected, the output shapes are inferred from the input parameters of the layer
/// and any mismatch is reported.
/// When InferAndValidate is selected 2 actions are performed: (1)infer output shape from inputs and (2)validate the
/// shapes as in ValidateOnly. This option has been added to work with tensors which rank or dimension sizes are not
/// specified explicitly, however this information can be calculated from the inputs.
///
enum class ShapeInferenceMethod
{
    /// Validate all output shapes
    ValidateOnly     = 0,
    /// Infer missing output shapes and validate all output shapes
    InferAndValidate = 1
};

/// Define the Memory Source to reduce copies
enum class MemorySource : uint32_t
{
    Undefined = 0,
    Malloc = 1,
    DmaBuf = 2,
    DmaBufProtected = 4,
    Gralloc = 5
};

enum class MemBlockStrategyType
{
    // MemBlocks can be packed on the Y axis only, overlap allowed on X axis.
    // In other words MemBlocks with overlapping lifetimes cannot use the same MemBin,
    // equivalent to blob or pooling memory management.
    SingleAxisPacking  = 0,

    // MemBlocks can be packed on either Y or X axis but cannot overlap on both.
    // In other words MemBlocks with overlapping lifetimes can use the same MemBin,
    // equivalent to offset or slab memory management.
    MultiAxisPacking  = 1
};

/// Each backend should implement an IBackend.
class IBackend
{
protected:
    IBackend() {}
    virtual ~IBackend() {}

public:
    virtual const BackendId& GetId() const = 0;
};

using IBackendSharedPtr = std::shared_ptr<IBackend>;
using IBackendUniquePtr = std::unique_ptr<IBackend, void(*)(IBackend* backend)>;

/// BackendCapability class
enum class BackendCapability : uint32_t
{
    /// Constant weights can be accessed through the descriptors,
    /// On the other hand, non-const weights can be accessed through inputs.
    NonConstWeights,

    /// Asynchronous Execution.
    AsyncExecution,

    // add new enum values here
};

/// Device specific knowledge to be passed to the optimizer.
class IDeviceSpec
{
protected:
    IDeviceSpec() {}
    virtual ~IDeviceSpec() {}
public:
    virtual const BackendIdSet& GetSupportedBackends() const = 0;
};

/// Type of identifiers for bindable layers (inputs, outputs).
using LayerBindingId = int;
using ImportedInputId = unsigned int;
using ImportedOutputId = unsigned int;


class PermutationVector
{
public:
    using ValueType = unsigned int;
    using SizeType = unsigned int;
    using ArrayType = std::array<ValueType, MaxNumOfTensorDimensions>;
    using ConstIterator = typename ArrayType::const_iterator;

    /// @param dimMappings - Indicates how to translate tensor elements from a given source into the target destination,
    /// when source and target potentially have different memory layouts.
    ///
    /// E.g. For a 4-d tensor laid out in a memory with the format (Batch Element, Height, Width, Channels),
    /// which is to be passed as an input to ArmNN, each source dimension is mapped to the corresponding
    /// ArmNN dimension. The Batch dimension remains the same (0 -> 0). The source Height dimension is mapped
    /// to the location of the ArmNN Height dimension (1 -> 2). Similar arguments are made for the Width and
    /// Channels (2 -> 3 and 3 -> 1). This will lead to @ref m_DimMappings pointing to the following array:
    /// [ 0, 2, 3, 1 ].
    ///
    /// Note that the mapping should be reversed if considering the case of ArmNN 4-d outputs (Batch Element,
    /// Channels, Height, Width) being written to a destination with the format mentioned above. We now have
    /// 0 -> 0, 2 -> 1, 3 -> 2, 1 -> 3, which, when reordered, lead to the following @ref m_DimMappings contents:
    /// [ 0, 3, 1, 2 ].
    ///
    PermutationVector(const ValueType *dimMappings, SizeType numDimMappings);

    PermutationVector(std::initializer_list<ValueType> dimMappings);

    ///
    /// Indexing method with out-of-bounds error checking for the m_DimMappings array.
    /// @param i - integer value corresponding to index of m_DimMappings array to retrieve element from.
    /// @return element at index i of m_DimMappings array.
    /// @throws InvalidArgumentException when indexing out-of-bounds index of m_DimMappings array.
    ///
    ValueType operator[](SizeType i) const
    {
        if (i >= GetSize())
        {
            throw InvalidArgumentException("Invalid indexing of PermutationVector of size " + std::to_string(GetSize())
                                            + " at location [" + std::to_string(i) + "].");
        }
        return m_DimMappings.at(i);
    }

    SizeType GetSize() const { return m_NumDimMappings; }

    ConstIterator begin() const { return m_DimMappings.begin(); }
    /**
     *
     * @return pointer one past the end of the number of mapping not the length of m_DimMappings.
     */
    ConstIterator end() const { return m_DimMappings.begin() + m_NumDimMappings; }

    bool IsEqual(const PermutationVector& other) const
    {
        if (m_NumDimMappings != other.m_NumDimMappings) return false;
        for (unsigned int i = 0; i < m_NumDimMappings; ++i)
        {
            if (m_DimMappings[i] != other.m_DimMappings[i]) return false;
        }
        return true;
    }

    bool IsInverse(const PermutationVector& other) const
    {
        bool isInverse = (GetSize() == other.GetSize());
        for (SizeType i = 0; isInverse && (i < GetSize()); ++i)
        {
            isInverse = (m_DimMappings[other.m_DimMappings[i]] == i);
        }
        return isInverse;
    }

private:
    ArrayType m_DimMappings;
    /// Number of valid entries in @ref m_DimMappings
    SizeType m_NumDimMappings;
};

class ITensorHandle;

/// Define the type of callback for the Debug layer to call
/// @param guid - guid of layer connected to the input of the Debug layer
/// @param slotIndex - index of the output slot connected to the input of the Debug layer
/// @param tensorHandle - TensorHandle for the input tensor to the Debug layer
using DebugCallbackFunction = std::function<void(LayerGuid guid, unsigned int slotIndex, ITensorHandle* tensorHandle)>;

/// Define a timer and associated inference ID for recording execution times
using HighResolutionClock = std::chrono::high_resolution_clock::time_point;
using InferenceTimingPair = std::pair<HighResolutionClock, HighResolutionClock>;


/// This list uses X macro technique.
/// See https://en.wikipedia.org/wiki/X_Macro for more info
#define LIST_OF_LAYER_TYPE \
    X(Activation) \
    X(Addition) \
    X(ArgMinMax) \
    X(BatchNormalization) \
    X(BatchToSpaceNd)      \
    X(Comparison) \
    X(Concat) \
    X(Constant) \
    X(ConvertBf16ToFp32) \
    X(ConvertFp16ToFp32) \
    X(ConvertFp32ToBf16) \
    X(ConvertFp32ToFp16) \
    X(Convolution2d) \
    X(Debug) \
    X(DepthToSpace) \
    X(DepthwiseConvolution2d) \
    X(Dequantize) \
    X(DetectionPostProcess) \
    X(Division) \
    X(ElementwiseUnary) \
    X(FakeQuantization) \
    X(Fill) \
    X(Floor) \
    X(FullyConnected) \
    X(Gather) \
    X(Input) \
    X(InstanceNormalization) \
    X(L2Normalization) \
    X(LogicalBinary) \
    X(LogSoftmax) \
    X(Lstm) \
    X(QLstm) \
    X(Map) \
    X(Maximum) \
    X(Mean) \
    X(MemCopy) \
    X(MemImport) \
    X(Merge) \
    X(Minimum) \
    X(Multiplication) \
    X(Normalization) \
    X(Output) \
    X(Pad) \
    X(Permute) \
    X(Pooling2d) \
    X(PreCompiled) \
    X(Prelu) \
    X(Quantize) \
    X(QuantizedLstm) \
    X(Reshape) \
    X(Rank) \
    X(Resize) \
    X(Reduce) \
    X(Slice) \
    X(Softmax) \
    X(SpaceToBatchNd) \
    X(SpaceToDepth) \
    X(Splitter) \
    X(Stack) \
    X(StandIn) \
    X(StridedSlice) \
    X(Subtraction) \
    X(Switch) \
    X(Transpose) \
    X(TransposeConvolution2d) \
    X(Unmap) \
    X(Cast) \
    X(Shape) \
    X(UnidirectionalSequenceLstm) \
    X(ChannelShuffle) \
    X(Convolution3d) \
    X(Pooling3d) \
    X(GatherNd) \
    X(BatchMatMul) \

// New layers should be added at last to minimize instability.

/// When adding a new layer, adapt also the LastLayer enum value in the
/// enum class LayerType below
enum class LayerType
{
#define X(name) name,
    LIST_OF_LAYER_TYPE
#undef X
    FirstLayer = Activation,
    LastLayer = UnidirectionalSequenceLstm
};

const char* GetLayerTypeAsCString(LayerType type);

} // namespace armnn
