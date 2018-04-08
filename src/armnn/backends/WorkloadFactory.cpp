//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "WorkloadFactory.hpp"
#include "RefWorkloadFactory.hpp"
#include "NeonWorkloadFactory.hpp"
#include "ClWorkloadFactory.hpp"

#include "armnn/Types.hpp"
#include "armnn/LayerSupport.hpp"
#include "Layer.hpp"
#include "Layers.hpp"
#include "CpuTensorHandle.hpp"

#include <boost/cast.hpp>
#include <cstring>
#include <boost/iterator/transform_iterator.hpp>

namespace armnn
{

bool IWorkloadFactory::IsLayerSupported(Compute compute, const Layer& layer, DataType dataType,
    std::string& outReasonIfUnsupported)
{
    constexpr size_t reasonCapacity = 1024;
    char reason[reasonCapacity];
    bool result;
    switch(layer.GetType())
    {
        case LayerType::Activation:
        {
            auto cLayer = boost::polymorphic_downcast<const ActivationLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsActivationSupported(compute, input, cLayer->GetParameters(), reason, reasonCapacity);
            break;
        }
        case LayerType::Addition:
        {
            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsAdditionSupported(compute, input0, input1, output, reason, reasonCapacity);
            break;
        }
        case LayerType::BatchNormalization:
        {
            auto cLayer = boost::polymorphic_downcast<const BatchNormalizationLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsBatchNormalizationSupported(compute, input, cLayer->GetParameters(), reason, reasonCapacity);
            break;
        }
        case LayerType::Constant:
        {
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsConstantSupported(compute, output, reason, reasonCapacity);
            break;
        }
        case LayerType::Convolution2d:
        {
            auto cLayer = boost::polymorphic_downcast<const Convolution2dLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsConvolution2dSupported(compute, input, cLayer->GetParameters(),
                                            cLayer->m_Weight->GetTensorInfo(), reason, reasonCapacity);
            break;
        }
        case LayerType::MemCopy:
        {
            // MemCopy supported for CpuRef, CpuAcc and GpuAcc backends
            // (also treat Undefined as CpuRef to avoid breaking lots of Unit tests)
            result = compute == Compute::CpuRef || compute == Compute::Undefined
                || compute == Compute::CpuAcc || compute == Compute::GpuAcc;
            strcpy(reason, "Unsupported backend type");
            break;
        }
        case LayerType::DepthwiseConvolution2d:
        {
            auto cLayer = boost::polymorphic_downcast<const DepthwiseConvolution2dLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsDepthwiseConvolutionSupported(compute, input, cLayer->GetParameters(),
                                                   cLayer->m_Weight->GetTensorInfo(), reason, reasonCapacity);
            break;
        }
        case LayerType::FakeQuantization:
        {
            auto cLayer = boost::polymorphic_downcast<const FakeQuantizationLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsFakeQuantizationSupported(compute, input, cLayer->GetParameters(), reason, reasonCapacity);
            break;
        }
        case LayerType::Floor:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsFloorSupported(compute, input, output, reason, reasonCapacity);
            break;
        }
        case LayerType::FullyConnected:
        {
            auto cLayer = boost::polymorphic_downcast<const FullyConnectedLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsFullyConnectedSupported(compute, input, cLayer->GetParameters(), reason, reasonCapacity);
            break;
        }
        case LayerType::Input:
        {
            const TensorInfo& input = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsInputSupported(compute, input, reason, reasonCapacity);
            break;
        }
        case LayerType::L2Normalization:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsL2NormalizationSupported(compute, input, reason, reasonCapacity);
            break;
        }
        case LayerType::Merger:
        {
            auto cLayer = boost::polymorphic_downcast<const MergerLayer*>(&layer);

            // Get vector of all inputs
            auto getTensorInfo = [](const InputSlot& slot)
                {
                    return &slot.GetConnectedOutputSlot()->GetTensorInfo();
                };
            auto begin = boost::make_transform_iterator(layer.GetInputSlots().begin(), getTensorInfo);
            auto end = boost::make_transform_iterator(layer.GetInputSlots().end(), getTensorInfo);

            std::vector<const TensorInfo*> inputs(begin, end);

            result = IsMergerSupported(compute, inputs, cLayer->GetParameters(), reason, reasonCapacity);
            break;
        }
        case LayerType::Multiplication:
        {
            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            result = IsMultiplicationSupported(compute, input0, input1, reason, reasonCapacity);
            break;
        }
        case LayerType::Normalization:
        {
            auto cLayer = boost::polymorphic_downcast<const NormalizationLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsNormalizationSupported(compute, input, output, cLayer->GetParameters(), reason, reasonCapacity);
            break;
        }
        case LayerType::Output:
        {
            const TensorInfo& output = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsOutputSupported(compute, output, reason, reasonCapacity);
            break;
        }
        case LayerType::Permute:
        {
            auto cLayer = boost::polymorphic_downcast<const PermuteLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsPermuteSupported(compute, input, output, cLayer->GetParameters(), reason, reasonCapacity);
            break;
        }
        case LayerType::Pooling2d:
        {
            auto cLayer = boost::polymorphic_downcast<const Pooling2dLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsPooling2dSupported(compute, input, output, cLayer->GetParameters(), reason, reasonCapacity);
            break;
        }
        case LayerType::Reshape:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsReshapeSupported(compute, input, reason, reasonCapacity);
            break;
        }
        case LayerType::ResizeBilinear:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsResizeBilinearSupported(compute, input, reason, reasonCapacity);
            break;
        }
        case LayerType::Softmax:
        {
            auto cLayer = boost::polymorphic_downcast<const SoftmaxLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsSoftmaxSupported(compute, input, cLayer->GetParameters(), reason, reasonCapacity);
            break;
        }
        case LayerType::Splitter:
        {
            auto cLayer = boost::polymorphic_downcast<const SplitterLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsSplitterSupported(compute, input, cLayer->GetParameters(), reason, reasonCapacity);
            break;
        }
        //
        case LayerType::DetectionOutput:
        {
            auto cLayer = boost::polymorphic_downcast<const DetectionOutputLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsSplitterSupported(compute, input, cLayer->GetParameters(), reason, reasonCapacity);
            break;
        }
        case LayerType::Reorg:
        {
            auto cLayer = boost::polymorphic_downcast<const ReorgLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsSplitterSupported(compute, input, cLayer->GetParameters(), reason, reasonCapacity);
            break;
        }
        default:
        {
            BOOST_ASSERT_MSG(false, "WorkloadFactory did not recognise type of layer.");
            strcpy(reason, "Unrecognised layer type");
            result = false;
            break;
        }
    }
    outReasonIfUnsupported = reason;
    return result;
}

bool IWorkloadFactory::IsLayerSupported(const Layer& layer, DataType dataType, std::string& outReasonIfUnsupported)
{
    return IsLayerSupported(layer.GetComputeDevice(), layer, dataType, outReasonIfUnsupported);
}

}