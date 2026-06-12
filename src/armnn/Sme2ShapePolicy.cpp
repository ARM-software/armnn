//
// Copyright © 2026 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Sme2ShapePolicy.hpp"

#include "Graph.hpp"
#include "Layer.hpp"
#include "armnnUtils/DataLayoutIndexed.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/Tensor.hpp>

#include <algorithm>
#include <cstdint>

namespace armnn
{
namespace
{

struct Sme2ShapeProfile
{
    unsigned int m_GemmLikeOps = 0;
    unsigned int m_DepthwiseConvolution2dOps = 0;
    unsigned int m_SmallDenseProjectionOps = 0;
    int64_t m_GemmMacs = 0;
    int64_t m_NonPointwiseGemmMacs = 0;
    bool m_HasFp16 = false;
    bool m_HasQuantized = false;
    bool m_HasSegmentationShape = false;
    bool m_HasStyleTransferShape = false;
    bool m_HasPoseShape = false;
    bool m_HasSmallMLargeNProjection = false;
};

bool IsQuantizedDataType(DataType dataType)
{
    switch (dataType)
    {
        case DataType::QAsymmU8:
        case DataType::QAsymmS8:
        case DataType::QSymmS8:
        case DataType::QSymmS16:
            return true;
        default:
            return false;
    }
}

void RecordTensorType(Sme2ShapeProfile& profile, const TensorInfo& tensorInfo)
{
    const DataType dataType = tensorInfo.GetDataType();
    profile.m_HasFp16 |= dataType == DataType::Float16;
    profile.m_HasQuantized |= IsQuantizedDataType(dataType);
}

bool HasSpecifiedShape(const TensorInfo& tensorInfo)
{
    const TensorShape& shape = tensorInfo.GetShape();
    return shape.GetDimensionality() == Dimensionality::Specified &&
           shape.AreAllDimensionsSpecified();
}

int64_t NumElements(const TensorShape& shape)
{
    if (shape.GetDimensionality() != Dimensionality::Specified ||
        !shape.AreAllDimensionsSpecified())
    {
        return 0;
    }

    int64_t elements = 1;
    for (unsigned int i = 0; i < shape.GetNumDimensions(); ++i)
    {
        elements *= static_cast<int64_t>(std::max(shape[i], 1U));
    }
    return elements;
}

int64_t Dimension(const TensorShape& shape, unsigned int index)
{
    if (shape.GetDimensionality() != Dimensionality::Specified ||
        !shape.AreAllDimensionsSpecified() ||
        index >= shape.GetNumDimensions())
    {
        return 0;
    }
    return static_cast<int64_t>(shape[index]);
}

int64_t DimensionFromEnd(const TensorShape& shape, unsigned int offset)
{
    if (offset == 0 || shape.GetNumDimensions() < offset)
    {
        return 0;
    }
    return Dimension(shape, shape.GetNumDimensions() - offset);
}

void RecordGemmShape(Sme2ShapeProfile& profile,
                     int64_t m,
                     int64_t n,
                     int64_t k,
                     int64_t kernelH,
                     int64_t kernelW,
                     bool isDenseProjection)
{
    if (m <= 0 || n <= 0 || k <= 0)
    {
        return;
    }

    ++profile.m_GemmLikeOps;

    const bool is1x1 = kernelH == 1 && kernelW == 1;
    const int64_t macs = m * n * k;
    profile.m_GemmMacs += macs;
    if (!is1x1)
    {
        profile.m_NonPointwiseGemmMacs += macs;
    }
    if (isDenseProjection && is1x1 && m <= 256 && n <= 1024 && k <= 1024)
    {
        ++profile.m_SmallDenseProjectionOps;
    }

    const bool hasModerateSpatialM = m >= 2048 && m <= 2560;
    if (is1x1 && hasModerateSpatialM && ((n >= 900 && k <= 384) || (n <= 384 && k >= 900)))
    {
        profile.m_HasSegmentationShape = true;
    }

    if (!is1x1 && ((m >= 25000 && m <= 30000 && n == 64 && k >= 2000) ||
                   (m >= 60000 && m <= 75000 && n == 32 && k >= 500)))
    {
        profile.m_HasStyleTransferShape = true;
    }

    if (!is1x1 && m >= 100000 && n >= 64 && k >= 500)
    {
        profile.m_HasPoseShape = true;
    }

    if (m <= 64 && n >= 4096 && k >= 64 && k <= 1024)
    {
        profile.m_HasSmallMLargeNProjection = true;
    }
}

void RecordConvolution2d(Sme2ShapeProfile& profile, const Layer& layer)
{
    if (layer.GetNumInputSlots() < 2 ||
        layer.GetNumOutputSlots() == 0 ||
        !layer.GetInputSlot(0).IsTensorInfoSet() ||
        !layer.GetInputSlot(1).IsTensorInfoSet() ||
        !layer.GetOutputSlot(0).IsTensorInfoSet())
    {
        return;
    }

    const TensorInfo& inputInfo = layer.GetInputSlot(0).GetTensorInfo();
    const TensorInfo& filterInfo = layer.GetInputSlot(1).GetTensorInfo();
    const TensorInfo& outputInfo = layer.GetOutputSlot(0).GetTensorInfo();
    RecordTensorType(profile, inputInfo);
    RecordTensorType(profile, filterInfo);
    RecordTensorType(profile, outputInfo);

    if (!HasSpecifiedShape(filterInfo) || !HasSpecifiedShape(outputInfo))
    {
        return;
    }

    const Convolution2dDescriptor& descriptor =
        static_cast<const Convolution2dDescriptor&>(layer.GetParameters());
    const TensorShape& filterShape = filterInfo.GetShape();
    const TensorShape& outputShape = outputInfo.GetShape();
    const armnnUtils::DataLayoutIndexed dataLayoutIndex(descriptor.m_DataLayout);

    if (filterShape.GetNumDimensions() != 4 || outputShape.GetNumDimensions() != 4)
    {
        return;
    }

    const int64_t n = Dimension(filterShape, 0);
    const int64_t kernelH = Dimension(filterShape, dataLayoutIndex.GetHeightIndex());
    const int64_t kernelW = Dimension(filterShape, dataLayoutIndex.GetWidthIndex());
    const int64_t filterElements = NumElements(filterShape);
    const int64_t inputChannels = n > 0 && kernelH > 0 && kernelW > 0 ?
        filterElements / (n * kernelH * kernelW) : 0;
    const int64_t k = kernelH * kernelW * inputChannels;
    const int64_t outputElements = NumElements(outputShape);
    const int64_t m = n > 0 ? outputElements / n : 0;

    RecordGemmShape(profile, m, n, k, kernelH, kernelW, false);
}

void RecordFullyConnected(Sme2ShapeProfile& profile, const Layer& layer)
{
    if (layer.GetNumInputSlots() < 2 ||
        layer.GetNumOutputSlots() == 0 ||
        !layer.GetInputSlot(0).IsTensorInfoSet() ||
        !layer.GetInputSlot(1).IsTensorInfoSet() ||
        !layer.GetOutputSlot(0).IsTensorInfoSet())
    {
        return;
    }

    const TensorInfo& inputInfo = layer.GetInputSlot(0).GetTensorInfo();
    const TensorInfo& weightsInfo = layer.GetInputSlot(1).GetTensorInfo();
    const TensorInfo& outputInfo = layer.GetOutputSlot(0).GetTensorInfo();
    RecordTensorType(profile, inputInfo);
    RecordTensorType(profile, weightsInfo);
    RecordTensorType(profile, outputInfo);

    if (!HasSpecifiedShape(inputInfo) || !HasSpecifiedShape(weightsInfo) || !HasSpecifiedShape(outputInfo))
    {
        return;
    }

    const TensorShape& weightsShape = weightsInfo.GetShape();
    if (weightsShape.GetNumDimensions() < 2)
    {
        return;
    }

    const FullyConnectedDescriptor& descriptor =
        static_cast<const FullyConnectedDescriptor&>(layer.GetParameters());
    const unsigned int nIndex = descriptor.m_TransposeWeightMatrix ? 0U : 1U;
    const unsigned int kIndex = descriptor.m_TransposeWeightMatrix ? 1U : 0U;
    const int64_t n = Dimension(weightsShape, nIndex);
    const int64_t k = Dimension(weightsShape, kIndex);
    const int64_t outputElements = NumElements(outputInfo.GetShape());
    const int64_t m = n > 0 ? outputElements / n : 0;

    RecordGemmShape(profile, m, n, k, 1, 1, true);
}

void RecordBatchMatMul(Sme2ShapeProfile& profile, const Layer& layer)
{
    if (layer.GetNumInputSlots() < 2 ||
        layer.GetNumOutputSlots() == 0 ||
        !layer.GetInputSlot(0).IsTensorInfoSet() ||
        !layer.GetInputSlot(1).IsTensorInfoSet() ||
        !layer.GetOutputSlot(0).IsTensorInfoSet())
    {
        return;
    }

    const TensorInfo& lhsInfo = layer.GetInputSlot(0).GetTensorInfo();
    const TensorInfo& rhsInfo = layer.GetInputSlot(1).GetTensorInfo();
    const TensorInfo& outputInfo = layer.GetOutputSlot(0).GetTensorInfo();
    RecordTensorType(profile, lhsInfo);
    RecordTensorType(profile, rhsInfo);
    RecordTensorType(profile, outputInfo);

    if (!HasSpecifiedShape(lhsInfo) || !HasSpecifiedShape(rhsInfo) || !HasSpecifiedShape(outputInfo))
    {
        return;
    }

    const TensorShape& lhsShape = lhsInfo.GetShape();
    const TensorShape& outputShape = outputInfo.GetShape();
    const int64_t n = DimensionFromEnd(outputShape, 1);
    const int64_t m = n > 0 ? NumElements(outputShape) / n : 0;
    int64_t k = DimensionFromEnd(lhsShape, 1);
    if (k == n)
    {
        k = DimensionFromEnd(lhsShape, 2);
    }

    RecordGemmShape(profile, m, n, k, 1, 1, true);
}

void RecordDepthwiseConvolution2d(Sme2ShapeProfile& profile, const Layer& layer)
{
    ++profile.m_DepthwiseConvolution2dOps;

    for (unsigned int i = 0; i < layer.GetNumInputSlots(); ++i)
    {
        if (layer.GetInputSlot(i).IsTensorInfoSet())
        {
            RecordTensorType(profile, layer.GetInputSlot(i).GetTensorInfo());
        }
    }
    for (unsigned int i = 0; i < layer.GetNumOutputSlots(); ++i)
    {
        if (layer.GetOutputSlot(i).IsTensorInfoSet())
        {
            RecordTensorType(profile, layer.GetOutputSlot(i).GetTensorInfo());
        }
    }
}

Sme2ShapeProfile BuildSme2ShapeProfile(const Graph& graph, bool reduceFp32ToFp16)
{
    Sme2ShapeProfile profile;
    profile.m_HasFp16 = reduceFp32ToFp16;

    for (const Layer* layer : graph)
    {
        switch (layer->GetType())
        {
            case LayerType::Convolution2d:
                RecordConvolution2d(profile, *layer);
                break;
            case LayerType::FullyConnected:
                RecordFullyConnected(profile, *layer);
                break;
            case LayerType::BatchMatMul:
                RecordBatchMatMul(profile, *layer);
                break;
            case LayerType::DepthwiseConvolution2d:
                RecordDepthwiseConvolution2d(profile, *layer);
                break;
            default:
                for (unsigned int i = 0; i < layer->GetNumInputSlots(); ++i)
                {
                    if (layer->GetInputSlot(i).IsTensorInfoSet())
                    {
                        RecordTensorType(profile, layer->GetInputSlot(i).GetTensorInfo());
                    }
                }
                for (unsigned int i = 0; i < layer->GetNumOutputSlots(); ++i)
                {
                    if (layer->GetOutputSlot(i).IsTensorInfoSet())
                    {
                        RecordTensorType(profile, layer->GetOutputSlot(i).GetTensorInfo());
                    }
                }
                break;
        }
    }

    return profile;
}

unsigned int CapWorkerCount(unsigned int workers, unsigned int cap)
{
    if (workers == 0 || cap == 0 || cap >= workers)
    {
        return workers;
    }
    return cap;
}

unsigned int GetCpuAccNumberOfThreads(const ModelOptions& modelOptions)
{
    unsigned int numberOfThreads = 0;
    ParseOptions(modelOptions, "CpuAcc", [&](std::string name, const BackendOptions::Var& value)
    {
        if (name == "NumberOfThreads")
        {
            if (value.IsUnsignedInt())
            {
                numberOfThreads = value.AsUnsignedInt();
            }
            else if (value.IsInt() && value.AsInt() > 0)
            {
                numberOfThreads = static_cast<unsigned int>(value.AsInt());
            }
        }
    });
    return numberOfThreads;
}

bool HasFloatSmeRegressionRisk(const Sme2ShapeProfile& profile)
{
    const bool isFloatOnly = !profile.m_HasFp16 && !profile.m_HasQuantized;
    if (!isFloatOnly)
    {
        return false;
    }

    const bool hasHeavySpatialConvolution =
        profile.m_GemmMacs > 0 &&
        profile.m_NonPointwiseGemmMacs * 2 >= profile.m_GemmMacs &&
        !profile.m_HasSegmentationShape;

    const bool hasSmallDenseGraph =
        profile.m_DepthwiseConvolution2dOps == 0 &&
        profile.m_SmallDenseProjectionOps >= 4 &&
        !profile.m_HasSmallMLargeNProjection;

    return profile.m_HasPoseShape ||
           profile.m_HasStyleTransferShape ||
           hasHeavySpatialConvolution ||
           hasSmallDenseGraph;
}

bool ShouldDisableSme(const Sme2ShapeProfile& profile)
{
    if (profile.m_GemmLikeOps == 0)
    {
        return false;
    }

    if (profile.m_HasFp16)
    {
        return true;
    }

    if (profile.m_HasQuantized)
    {
        return !profile.m_HasSmallMLargeNProjection;
    }

    return HasFloatSmeRegressionRisk(profile);
}

unsigned int SelectNumberOfThreads(const Sme2ShapeProfile& profile, unsigned int requestedThreads)
{
    if (!profile.m_HasQuantized || ShouldDisableSme(profile))
    {
        return requestedThreads;
    }

    if (profile.m_GemmLikeOps == 0)
    {
        return CapWorkerCount(requestedThreads, 1);
    }

    if (profile.m_HasSegmentationShape || profile.m_HasStyleTransferShape)
    {
        return requestedThreads;
    }

    if (profile.m_HasPoseShape)
    {
        return CapWorkerCount(requestedThreads, 4);
    }

    return CapWorkerCount(requestedThreads, 1);
}

} // namespace

void ApplySme2ShapePolicy(const Graph& graph, bool reduceFp32ToFp16, ModelOptions& modelOptions)
{
    const Sme2ShapeProfile profile = BuildSme2ShapeProfile(graph, reduceFp32ToFp16);
    if (profile.m_GemmLikeOps == 0)
    {
        return;
    }

    const bool smeEnabled = !ShouldDisableSme(profile);
    const bool sveEnabled = smeEnabled || profile.m_HasQuantized;
    const unsigned int requestedThreads = GetCpuAccNumberOfThreads(modelOptions);
    const unsigned int selectedThreads = SelectNumberOfThreads(profile, requestedThreads);

    modelOptions.push_back(BackendOptions("CpuAcc", {{"SmeEnabled", smeEnabled}, {"SveEnabled", sveEnabled}}));
    if (selectedThreads != requestedThreads)
    {
        modelOptions.push_back(BackendOptions("CpuAcc", {{"NumberOfThreads", selectedThreads}}));
    }
}

} // namespace armnn
