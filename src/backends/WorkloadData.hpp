//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "WorkloadDataFwd.hpp"

#include "armnn/Types.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/Descriptors.hpp"
#include "armnn/Exceptions.hpp"
#include "InternalTypes.hpp"
#include "OutputHandler.hpp"
#include "CpuTensorHandleFwd.hpp"

namespace armnn
{

//A helper function that returns the bias data type required for given input data type.
DataType GetBiasDataType(DataType inputDataType);

struct WorkloadInfo;

struct QueueDescriptor
{
    std::vector<ITensorHandle*> m_Inputs;
    std::vector<ITensorHandle*> m_Outputs;

    void ValidateInputsOutputs(const std::string& descName,
        unsigned int numExpectedIn, unsigned int numExpectedOut) const;


protected:
    ~QueueDescriptor() = default;
    QueueDescriptor() = default;
    QueueDescriptor(QueueDescriptor const&) = default;
    QueueDescriptor& operator=(QueueDescriptor const&) = default;
};

// Base class for queue descriptors which contain parameters.
template <typename LayerDescriptor>
struct QueueDescriptorWithParameters : public QueueDescriptor
{
    LayerDescriptor m_Parameters;

protected:
    ~QueueDescriptorWithParameters() = default;
    QueueDescriptorWithParameters() = default;
    QueueDescriptorWithParameters(QueueDescriptorWithParameters const&) = default;
    QueueDescriptorWithParameters& operator=(QueueDescriptorWithParameters const&) = default;
};

struct MemCopyQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

using InputQueueDescriptor = MemCopyQueueDescriptor;
using OutputQueueDescriptor = MemCopyQueueDescriptor;

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

// Merger layer workload data.
struct MergerQueueDescriptor : QueueDescriptorWithParameters<OriginsDescriptor>
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

// Activation layer workload data.
struct ActivationQueueDescriptor : QueueDescriptorWithParameters<ActivationDescriptor>
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Fully connected layer workload data.
struct FullyConnectedQueueDescriptor : QueueDescriptorWithParameters<FullyConnectedDescriptor>
{
    FullyConnectedQueueDescriptor()
        : m_Weight(nullptr)
        , m_Bias(nullptr)
    {
    }

    const ConstCpuTensorHandle* m_Weight;
    const ConstCpuTensorHandle* m_Bias;

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

// Convolution 2D layer workload data.
struct Convolution2dQueueDescriptor : QueueDescriptorWithParameters<Convolution2dDescriptor>
{
    Convolution2dQueueDescriptor()
        : m_Weight(nullptr)
        , m_Bias(nullptr)
        , m_DataLayout(DataLayout::NCHW)
    {
    }

    const ConstCpuTensorHandle* m_Weight;
    const ConstCpuTensorHandle* m_Bias;
    DataLayout m_DataLayout;

    void Validate(const WorkloadInfo& workloadInfo) const;
};

// Depthwise Convolution 2D layer workload data.
struct DepthwiseConvolution2dQueueDescriptor : QueueDescriptorWithParameters<DepthwiseConvolution2dDescriptor>
{
    DepthwiseConvolution2dQueueDescriptor()
        : m_Weight(nullptr)
        , m_Bias(nullptr)
    {
    }

    const ConstCpuTensorHandle* m_Weight;
    const ConstCpuTensorHandle* m_Bias;

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

    const ConstCpuTensorHandle* m_Mean;
    const ConstCpuTensorHandle* m_Variance;
    const ConstCpuTensorHandle* m_Beta;
    const ConstCpuTensorHandle* m_Gamma;

    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct ResizeBilinearQueueDescriptor : QueueDescriptorWithParameters<ResizeBilinearDescriptor>
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

    const ConstCpuTensorHandle* m_Min;
    const ConstCpuTensorHandle* m_Max;

    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct L2NormalizationQueueDescriptor : QueueDescriptor
{
    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct ConstantQueueDescriptor : QueueDescriptor
{
    ConstantQueueDescriptor()
        : m_LayerOutput(nullptr)
    {
    }

    const ConstCpuTensorHandle* m_LayerOutput;

    void Validate(const WorkloadInfo& workloadInfo) const;
};

struct ReshapeQueueDescriptor : QueueDescriptorWithParameters<ReshapeDescriptor>
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
    {
    }

    const ConstCpuTensorHandle* m_InputToInputWeights;
    const ConstCpuTensorHandle* m_InputToForgetWeights;
    const ConstCpuTensorHandle* m_InputToCellWeights;
    const ConstCpuTensorHandle* m_InputToOutputWeights;
    const ConstCpuTensorHandle* m_RecurrentToInputWeights;
    const ConstCpuTensorHandle* m_RecurrentToForgetWeights;
    const ConstCpuTensorHandle* m_RecurrentToCellWeights;
    const ConstCpuTensorHandle* m_RecurrentToOutputWeights;
    const ConstCpuTensorHandle* m_CellToInputWeights;
    const ConstCpuTensorHandle* m_CellToForgetWeights;
    const ConstCpuTensorHandle* m_CellToOutputWeights;
    const ConstCpuTensorHandle* m_InputGateBias;
    const ConstCpuTensorHandle* m_ForgetGateBias;
    const ConstCpuTensorHandle* m_CellBias;
    const ConstCpuTensorHandle* m_OutputGateBias;
    const ConstCpuTensorHandle* m_ProjectionWeights;
    const ConstCpuTensorHandle* m_ProjectionBias;

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

} //namespace armnn
