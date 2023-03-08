//
// Copyright © 2021-2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "TensorHandle.hpp"

#include <armnn/Deprecated.hpp>
#include <armnn/Descriptors.hpp>
#include <armnn/Exceptions.hpp>
#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>
#include <common/include/ProfilingGuid.hpp>

namespace armnn
{

//A helper function that returns the bias data type required for given input data type.
DataType GetBiasDataType(DataType inputDataType);

struct WorkloadInfo;

struct QueueDescriptor
{
    std::vector<ITensorHandle*> m_Inputs;
    std::vector<ITensorHandle*> m_Outputs;
    void* m_AdditionalInfoObject;

    virtual ~QueueDescriptor() = default;

    void ValidateTensorNumDimensions(const TensorInfo& tensor,
                                     std::string const& descName,
                                     unsigned int numDimensions,
                                     std::string const& tensorName) const;

    void ValidateTensorNumDimNumElem(const TensorInfo& tensorInfo,
                                     unsigned int numDimension,
                                     unsigned int numElements,
                                     std::string const& tensorName) const;

    void ValidateInputsOutputs(const std::string& descName,
                               unsigned int numExpectedIn,
                               unsigned int numExpectedOut) const;

    template<typename T>
    const T* GetAdditionalInformation() const
    {
        return static_cast<T*>(m_AdditionalInfoObject);
    }

    bool m_AllowExpandedDims = false;

protected:
    QueueDescriptor()
        : m_AdditionalInfoObject(nullptr)
    {}
    QueueDescriptor(QueueDescriptor const&) = default;
    QueueDescriptor& operator=(QueueDescriptor const&) = default;
};

// Base class for queue descriptors which contain parameters.
template <typename LayerDescriptor>
struct QueueDescriptorWithParameters : public QueueDescriptor
{
    LayerDescriptor m_Parameters;

    virtual ~QueueDescriptorWithParameters() = default;

protected:
    QueueDescriptorWithParameters() = default;
    QueueDescriptorWithParameters(QueueDescriptorWithParameters const&) = default;
    QueueDescriptorWithParameters& operator=(QueueDescriptorWithParameters const&) = default;
};

struct MapQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct UnmapQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct MemCopyQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

using InputQueueDescriptor = MemCopyQueueDescriptor;
using OutputQueueDescriptor = MemCopyQueueDescriptor;

struct MemImportQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct MemSyncQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Softmax layer workload data.
struct SoftmaxQueueDescriptor : QueueDescriptorWithParameters<SoftmaxDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Splitter layer workload data.
struct SplitterQueueDescriptor : QueueDescriptorWithParameters<ViewsDescriptor>
{
    struct ViewOrigin
    {
        ViewOrigin() {}
        ViewOrigin(std::vector<unsigned int> const& origin) : m_Origin(origin) {}

        //View origin (size of the vector is the same as number of dimensions of the view).
        std::vector<unsigned int> m_Origin;
    };

    //View defines a tensor that will be carved from the input tensor.
    //View origins are stored here, the extents are defined by sizes of the output tensors.
    std::vector<ViewOrigin> m_ViewOrigins;

    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Concat layer workload data.
struct ConcatQueueDescriptor : QueueDescriptorWithParameters<OriginsDescriptor>
{
    struct ViewOrigin
    {
        ViewOrigin() {}
        ViewOrigin(const std::vector<unsigned int>& origin) : m_Origin(origin) {}

        //View origin (size of the vector is the same as number of dimensions of the view).
        std::vector<unsigned int> m_Origin;
    };

    //View defines a sub-area of the output tensor that will be filled with the corresponding input tensor.
    //View origins are stored here, the extents are defined by sizes of the input tensors.
    std::vector<ViewOrigin> m_ViewOrigins;

    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Deprecated. Use ConcatQueueDescriptor instead
using MergerQueueDescriptor = ConcatQueueDescriptor;

// Stack layer workload data.
struct StackQueueDescriptor : QueueDescriptorWithParameters<StackDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Activation layer workload data.
struct ActivationQueueDescriptor : QueueDescriptorWithParameters<ActivationDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct ArgMinMaxQueueDescriptor : QueueDescriptorWithParameters<ArgMinMaxDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct CastQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Fill layer workload data.
struct FillQueueDescriptor : QueueDescriptorWithParameters<FillDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Fully connected layer workload data.
struct FullyConnectedQueueDescriptor : QueueDescriptorWithParameters<FullyConnectedDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Permute layer workload data.
struct PermuteQueueDescriptor : QueueDescriptorWithParameters<PermuteDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Pooling 2D layer workload data.
struct Pooling2dQueueDescriptor : QueueDescriptorWithParameters<Pooling2dDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Pooling 3D layer workload data.
struct Pooling3dQueueDescriptor : QueueDescriptorWithParameters<Pooling3dDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};


// Convolution 2D layer workload data.
struct Convolution2dQueueDescriptor : QueueDescriptorWithParameters<Convolution2dDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Convolution 3D layer workload data.
struct Convolution3dQueueDescriptor : QueueDescriptorWithParameters<Convolution3dDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

/// Depthwise Convolution 2D layer workload data.
///
/// @note
/// The weights are in the format [1, H, W, I*M]. Where I is the input channel size, M the depthwise mutliplier and
/// H, W is the height and width of the filter kernel. If per channel quantization is applied
/// the weights will be quantized along the last dimension/axis (I*M) which corresponds to the output channel size.
/// If per channel quantization is applied the weights tensor will have I*M scales, one for each dimension
/// of the quantization axis. You have to be aware of this when reshaping the weights tensor.
/// Splitting the I*M axis, e.g. [1, H, W, I*M] --> [H, W, I, M], won't work without taking care of the
/// corresponding quantization scales.
/// If there is no per channel quantization applied reshaping the weights tensor won't cause any issues. There are
/// preconfigured permutation functions available @link WorkloadUtils.hpp here.
///
struct DepthwiseConvolution2dQueueDescriptor : QueueDescriptorWithParameters<DepthwiseConvolution2dDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct DetectionPostProcessQueueDescriptor : QueueDescriptorWithParameters<DetectionPostProcessDescriptor>
{
    DetectionPostProcessQueueDescriptor()
        : m_Anchors(nullptr)
    {
    }

    const ConstTensorHandle* m_Anchors;

    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Normalization layer workload data.
struct NormalizationQueueDescriptor : QueueDescriptorWithParameters<NormalizationDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Add layer workload data.
struct AdditionQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Multiplication layer workload data.
struct MultiplicationQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Division layer workload data.
struct DivisionQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Subtraction layer workload data.
struct SubtractionQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Maximum layer workload data.
struct MaximumQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Mean layer workload data.
struct MeanQueueDescriptor : QueueDescriptorWithParameters<MeanDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Pad layer workload data
struct PadQueueDescriptor : QueueDescriptorWithParameters<PadDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct QuantizeQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Deprecated use ComparisonQueueDescriptor instead
struct EqualQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Batch norm layer workload data.
struct BatchNormalizationQueueDescriptor : QueueDescriptorWithParameters<BatchNormalizationDescriptor>
{
    BatchNormalizationQueueDescriptor()
        : m_Mean(nullptr)
        , m_Variance(nullptr)
        , m_Beta(nullptr)
        , m_Gamma(nullptr)
    {
    }

    const ConstTensorHandle* m_Mean;
    const ConstTensorHandle* m_Variance;
    const ConstTensorHandle* m_Beta;
    const ConstTensorHandle* m_Gamma;

    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct RankQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct ResizeQueueDescriptor : QueueDescriptorWithParameters<ResizeDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct FakeQuantizationQueueDescriptor : QueueDescriptorWithParameters<FakeQuantizationDescriptor>
{
    FakeQuantizationQueueDescriptor()
    : m_Min(nullptr)
    , m_Max(nullptr)
    {
    }

    const ConstTensorHandle* m_Min;
    const ConstTensorHandle* m_Max;

    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct InstanceNormalizationQueueDescriptor : QueueDescriptorWithParameters<InstanceNormalizationDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct L2NormalizationQueueDescriptor : QueueDescriptorWithParameters<L2NormalizationDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct LogSoftmaxQueueDescriptor : QueueDescriptorWithParameters<LogSoftmaxDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct ConstantQueueDescriptor : QueueDescriptor
{
    ConstantQueueDescriptor()
        : m_LayerOutput(nullptr)
    {
    }

    const ConstTensorHandle* m_LayerOutput;

    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct ReshapeQueueDescriptor : QueueDescriptorWithParameters<ReshapeDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct SpaceToBatchNdQueueDescriptor : QueueDescriptorWithParameters<SpaceToBatchNdDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct SpaceToDepthQueueDescriptor : QueueDescriptorWithParameters<SpaceToDepthDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct FloorQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct LstmQueueDescriptor : QueueDescriptorWithParameters<LstmDescriptor>
{
    LstmQueueDescriptor()
        : m_InputToInputWeights(nullptr)
        , m_InputToForgetWeights(nullptr)
        , m_InputToCellWeights(nullptr)
        , m_InputToOutputWeights(nullptr)
        , m_RecurrentToInputWeights(nullptr)
        , m_RecurrentToForgetWeights(nullptr)
        , m_RecurrentToCellWeights(nullptr)
        , m_RecurrentToOutputWeights(nullptr)
        , m_CellToInputWeights(nullptr)
        , m_CellToForgetWeights(nullptr)
        , m_CellToOutputWeights(nullptr)
        , m_InputGateBias(nullptr)
        , m_ForgetGateBias(nullptr)
        , m_CellBias(nullptr)
        , m_OutputGateBias(nullptr)
        , m_ProjectionWeights(nullptr)
        , m_ProjectionBias(nullptr)
        , m_InputLayerNormWeights(nullptr)
        , m_ForgetLayerNormWeights(nullptr)
        , m_CellLayerNormWeights(nullptr)
        , m_OutputLayerNormWeights(nullptr)
    {
    }

    const ConstTensorHandle* m_InputToInputWeights;
    const ConstTensorHandle* m_InputToForgetWeights;
    const ConstTensorHandle* m_InputToCellWeights;
    const ConstTensorHandle* m_InputToOutputWeights;
    const ConstTensorHandle* m_RecurrentToInputWeights;
    const ConstTensorHandle* m_RecurrentToForgetWeights;
    const ConstTensorHandle* m_RecurrentToCellWeights;
    const ConstTensorHandle* m_RecurrentToOutputWeights;
    const ConstTensorHandle* m_CellToInputWeights;
    const ConstTensorHandle* m_CellToForgetWeights;
    const ConstTensorHandle* m_CellToOutputWeights;
    const ConstTensorHandle* m_InputGateBias;
    const ConstTensorHandle* m_ForgetGateBias;
    const ConstTensorHandle* m_CellBias;
    const ConstTensorHandle* m_OutputGateBias;
    const ConstTensorHandle* m_ProjectionWeights;
    const ConstTensorHandle* m_ProjectionBias;
    const ConstTensorHandle* m_InputLayerNormWeights;
    const ConstTensorHandle* m_ForgetLayerNormWeights;
    const ConstTensorHandle* m_CellLayerNormWeights;
    const ConstTensorHandle* m_OutputLayerNormWeights;

    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct ConvertFp16ToFp32QueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct ConvertFp32ToFp16QueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct BatchToSpaceNdQueueDescriptor : QueueDescriptorWithParameters<BatchToSpaceNdDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct StridedSliceQueueDescriptor : QueueDescriptorWithParameters<StridedSliceDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Minimum layer workload data.
struct MinimumQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Deprecated use ComparisonQueueDescriptor instead
struct GreaterQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct DebugQueueDescriptor : QueueDescriptor
{
    DebugQueueDescriptor() : m_Guid(0) {}

    void Validate(const WorkloadInfo& workloadInfo) const;

    LayerGuid m_Guid;
    std::string m_LayerName;
    unsigned int m_SlotIndex;

    bool m_LayerOutputToFile = false;
};

struct RsqrtQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct GatherNdQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct GatherQueueDescriptor : QueueDescriptorWithParameters<GatherDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct PreCompiledQueueDescriptor : QueueDescriptorWithParameters<PreCompiledDescriptor>
{
    PreCompiledQueueDescriptor()
        : m_PreCompiledObject(nullptr)
    {
    }

    void* m_PreCompiledObject;

    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct DequantizeQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct MergeQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct SwitchQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct PreluQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct TransposeConvolution2dQueueDescriptor : QueueDescriptorWithParameters<TransposeConvolution2dDescriptor>
{
    TransposeConvolution2dQueueDescriptor() :
        m_Weight(nullptr),
        m_Bias(nullptr)
    {}

    const ConstTensorHandle* m_Weight;
    const ConstTensorHandle* m_Bias;

    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct TransposeQueueDescriptor : QueueDescriptorWithParameters<TransposeDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct QLstmQueueDescriptor : QueueDescriptorWithParameters<QLstmDescriptor>
{
    QLstmQueueDescriptor()
            : m_InputToInputWeights(nullptr)
            , m_InputToForgetWeights(nullptr)
            , m_InputToCellWeights(nullptr)
            , m_InputToOutputWeights(nullptr)
            , m_RecurrentToInputWeights(nullptr)
            , m_RecurrentToForgetWeights(nullptr)
            , m_RecurrentToCellWeights(nullptr)
            , m_RecurrentToOutputWeights(nullptr)
            , m_CellToInputWeights(nullptr)
            , m_CellToForgetWeights(nullptr)
            , m_CellToOutputWeights(nullptr)
            , m_InputGateBias(nullptr)
            , m_ForgetGateBias(nullptr)
            , m_CellBias(nullptr)
            , m_OutputGateBias(nullptr)
            , m_ProjectionWeights(nullptr)
            , m_ProjectionBias(nullptr)
            , m_InputLayerNormWeights(nullptr)
            , m_ForgetLayerNormWeights(nullptr)
            , m_CellLayerNormWeights(nullptr)
            , m_OutputLayerNormWeights(nullptr)
    {
    }

    const ConstTensorHandle* m_InputToInputWeights;
    const ConstTensorHandle* m_InputToForgetWeights;
    const ConstTensorHandle* m_InputToCellWeights;
    const ConstTensorHandle* m_InputToOutputWeights;
    const ConstTensorHandle* m_RecurrentToInputWeights;
    const ConstTensorHandle* m_RecurrentToForgetWeights;
    const ConstTensorHandle* m_RecurrentToCellWeights;
    const ConstTensorHandle* m_RecurrentToOutputWeights;
    const ConstTensorHandle* m_CellToInputWeights;
    const ConstTensorHandle* m_CellToForgetWeights;
    const ConstTensorHandle* m_CellToOutputWeights;
    const ConstTensorHandle* m_InputGateBias;
    const ConstTensorHandle* m_ForgetGateBias;
    const ConstTensorHandle* m_CellBias;
    const ConstTensorHandle* m_OutputGateBias;
    const ConstTensorHandle* m_ProjectionWeights;
    const ConstTensorHandle* m_ProjectionBias;
    const ConstTensorHandle* m_InputLayerNormWeights;
    const ConstTensorHandle* m_ForgetLayerNormWeights;
    const ConstTensorHandle* m_CellLayerNormWeights;
    const ConstTensorHandle* m_OutputLayerNormWeights;

    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct QuantizedLstmQueueDescriptor : QueueDescriptor
{
    QuantizedLstmQueueDescriptor()
        : m_InputToInputWeights(nullptr)
        , m_InputToForgetWeights(nullptr)
        , m_InputToCellWeights(nullptr)
        , m_InputToOutputWeights(nullptr)

        , m_RecurrentToInputWeights(nullptr)
        , m_RecurrentToForgetWeights(nullptr)
        , m_RecurrentToCellWeights(nullptr)
        , m_RecurrentToOutputWeights(nullptr)

        , m_InputGateBias(nullptr)
        , m_ForgetGateBias(nullptr)
        , m_CellBias(nullptr)
        , m_OutputGateBias(nullptr)
    {}

    const ConstTensorHandle* m_InputToInputWeights;
    const ConstTensorHandle* m_InputToForgetWeights;
    const ConstTensorHandle* m_InputToCellWeights;
    const ConstTensorHandle* m_InputToOutputWeights;

    const ConstTensorHandle* m_RecurrentToInputWeights;
    const ConstTensorHandle* m_RecurrentToForgetWeights;
    const ConstTensorHandle* m_RecurrentToCellWeights;
    const ConstTensorHandle* m_RecurrentToOutputWeights;

    const ConstTensorHandle* m_InputGateBias;
    const ConstTensorHandle* m_ForgetGateBias;
    const ConstTensorHandle* m_CellBias;
    const ConstTensorHandle* m_OutputGateBias;

    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct AbsQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct SliceQueueDescriptor : QueueDescriptorWithParameters<SliceDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct DepthToSpaceQueueDescriptor : QueueDescriptorWithParameters<DepthToSpaceDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct ComparisonQueueDescriptor : QueueDescriptorWithParameters<ComparisonDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct ElementwiseBinaryQueueDescriptor : QueueDescriptorWithParameters<ElementwiseBinaryDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct ElementwiseUnaryQueueDescriptor : QueueDescriptorWithParameters<ElementwiseUnaryDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct LogicalBinaryQueueDescriptor : QueueDescriptorWithParameters<LogicalBinaryDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct ReduceQueueDescriptor : QueueDescriptorWithParameters<ReduceDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct ShapeQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct UnidirectionalSequenceLstmQueueDescriptor : QueueDescriptorWithParameters<LstmDescriptor>
{
    UnidirectionalSequenceLstmQueueDescriptor()
        : m_InputToInputWeights(nullptr)
        , m_InputToForgetWeights(nullptr)
        , m_InputToCellWeights(nullptr)
        , m_InputToOutputWeights(nullptr)
        , m_RecurrentToInputWeights(nullptr)
        , m_RecurrentToForgetWeights(nullptr)
        , m_RecurrentToCellWeights(nullptr)
        , m_RecurrentToOutputWeights(nullptr)
        , m_CellToInputWeights(nullptr)
        , m_CellToForgetWeights(nullptr)
        , m_CellToOutputWeights(nullptr)
        , m_InputGateBias(nullptr)
        , m_ForgetGateBias(nullptr)
        , m_CellBias(nullptr)
        , m_OutputGateBias(nullptr)
        , m_ProjectionWeights(nullptr)
        , m_ProjectionBias(nullptr)
        , m_InputLayerNormWeights(nullptr)
        , m_ForgetLayerNormWeights(nullptr)
        , m_CellLayerNormWeights(nullptr)
        , m_OutputLayerNormWeights(nullptr)
    {
    }

    const ConstTensorHandle* m_InputToInputWeights;
    const ConstTensorHandle* m_InputToForgetWeights;
    const ConstTensorHandle* m_InputToCellWeights;
    const ConstTensorHandle* m_InputToOutputWeights;
    const ConstTensorHandle* m_RecurrentToInputWeights;
    const ConstTensorHandle* m_RecurrentToForgetWeights;
    const ConstTensorHandle* m_RecurrentToCellWeights;
    const ConstTensorHandle* m_RecurrentToOutputWeights;
    const ConstTensorHandle* m_CellToInputWeights;
    const ConstTensorHandle* m_CellToForgetWeights;
    const ConstTensorHandle* m_CellToOutputWeights;
    const ConstTensorHandle* m_InputGateBias;
    const ConstTensorHandle* m_ForgetGateBias;
    const ConstTensorHandle* m_CellBias;
    const ConstTensorHandle* m_OutputGateBias;
    const ConstTensorHandle* m_ProjectionWeights;
    const ConstTensorHandle* m_ProjectionBias;
    const ConstTensorHandle* m_InputLayerNormWeights;
    const ConstTensorHandle* m_ForgetLayerNormWeights;
    const ConstTensorHandle* m_CellLayerNormWeights;
    const ConstTensorHandle* m_OutputLayerNormWeights;

    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct ChannelShuffleQueueDescriptor : QueueDescriptorWithParameters<ChannelShuffleDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct BatchMatMulQueueDescriptor : QueueDescriptorWithParameters<BatchMatMulDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

} // namespace armnn
