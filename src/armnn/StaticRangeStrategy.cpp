//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "StaticRangeStrategy.hpp"

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/Descriptors.hpp>
#include <armnn/Types.hpp>

#include <limits>

namespace armnn
{

StaticRangeStrategy::StaticRangeStrategy(RangeTracker& rangeTracker)
    : m_RangeTracker(rangeTracker)
{}

void StaticRangeStrategy::SetRange(const IConnectableLayer* layer, unsigned int outputIdx, float min, float max)
{
    m_RangeTracker.SetRange(layer, outputIdx, min, max);
}

void StaticRangeStrategy::ForwardParentParameters(const IConnectableLayer* layer)
{
    const auto parentRange = m_RangeTracker.GetRange(layer->GetInputSlot(0).GetConnection()->GetOwningLayerGuid(), 0);
    SetRange(layer, 0, parentRange.first, parentRange.second);
}


void StaticRangeStrategy::ExecuteStrategy(const armnn::IConnectableLayer *layer,
                                          const BaseDescriptor &descriptor,
                                          const std::vector<armnn::ConstTensor> &constants,
                                          const char *name,
                                          const armnn::LayerBindingId id)
{
IgnoreUnused(id, name);

switch (layer->GetType())
{
    case armnn::LayerType::Activation :
    {
        const ActivationDescriptor& activationDescriptor = static_cast<const ActivationDescriptor&>(descriptor);

        switch (activationDescriptor.m_Function)
        {
            // Range is 0, 15 for Abs, Linear, ReLu and Soft ReLu
            case ActivationFunction::Abs:
            case ActivationFunction::Linear:
            case ActivationFunction::ReLu:
            case ActivationFunction::SoftReLu:
                SetRange(layer, 0, 0.f, 15.f);
                break;
            case ActivationFunction::BoundedReLu:
                SetRange(layer, 0, 0.f, activationDescriptor.m_A);
                break;
            case ActivationFunction::TanH:
                SetRange(layer, 0, -1.f, 1.f);
                break;
            case ActivationFunction::LeakyReLu:
                SetRange(layer, 0, -5.f, 15.f);
                break;
            default:
                SetRange(layer, 0, -15.f, 15.f);
                break;
        }
        break;
    }
    case armnn::LayerType::Addition :
    {
        SetRange(layer, 0, -20.f, 20.f);
        break;
    }
    case armnn::LayerType::ArgMinMax :
    {
        ForwardParentParameters(layer);
        break;
    }
    case armnn::LayerType::BatchToSpaceNd :
    {
        ForwardParentParameters(layer);
        break;
    }
    case armnn::LayerType::BatchNormalization :
    {
        SetRange(layer, 0, -15.0f, 15.0f);
        break;
    }
    case armnn::LayerType::Concat :
    {
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::lowest();
        for (unsigned int i = 0; i < layer->GetNumInputSlots(); ++i)
        {
            const IOutputSlot* outputSlot = layer->GetInputSlot(i).GetConnection();
            LayerGuid layerId = outputSlot->GetOwningLayerGuid();
            unsigned int slotIndex = outputSlot->CalculateIndexOnOwner();
            RangeTracker::MinMaxRange range = m_RangeTracker.GetRange(layerId, slotIndex);
            min = std::min(min, range.first);
            max = std::max(max, range.second);
        }
        SetRange(layer, 0, min, max);
        break;
    }
    case armnn::LayerType::Constant :
    {

        if (constants[0].GetDataType() != DataType::Float32)
        {
            throw InvalidArgumentException("Quantization is supported only for FP32 tensors");
        }

        // Work out the range based on the input constants
        unsigned int inputNumElements = constants[0].GetNumElements();
        const float* inputData = reinterpret_cast<const float*>(constants[0].GetMemoryArea());

        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::lowest();

        for (unsigned int i = 0; i < inputNumElements; i++)
        {
            const float inputValue = inputData[i];

            min = std::min(min, inputValue);
            max = std::max(max, inputValue);
        }
        SetRange(layer, 0, min, max);
        break;
    }
    case armnn::LayerType::Convolution2d :
    {
        SetRange(layer, 0, -15.0f, 15.0f);
        break;
    }
    case armnn::LayerType::DepthwiseConvolution2d :
    {
        SetRange(layer, 0, -15.0f, 15.0f);
        break;
    }
    case armnn::LayerType::FullyConnected :
    {
        SetRange(layer, 0, -15.0f, 15.0f);
        break;
    }
    case armnn::LayerType::Permute :
    {
        ForwardParentParameters(layer);
        break;
    }
    case armnn::LayerType::Pooling2d :
    {
        ForwardParentParameters(layer);
        break;
    }
    case armnn::LayerType::Reshape :
    {
        ForwardParentParameters(layer);
        break;
    }
    case armnn::LayerType::Resize :
    {
        ForwardParentParameters(layer);
        break;
    }
    case armnn::LayerType::Splitter :
    {
        ForwardParentParameters(layer);
        break;
    }
    case armnn::LayerType::SpaceToBatchNd :
    {
        ForwardParentParameters(layer);
        break;
    }
    case armnn::LayerType::Softmax :
    {
        SetRange(layer, 0, 0.f, 1.f);
        break;
    }
    case armnn::LayerType::StridedSlice :
    {
        ForwardParentParameters(layer);
        break;
    }
    default:
    {
    }
}
}

} //namespace armnn
