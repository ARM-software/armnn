//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DynamicQuantizationStrategy.hpp"
#include "NetworkUtils.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/Types.hpp>

#include <limits>

namespace armnn
{
DynamicQuantizationStrategy::DynamicQuantizationStrategy(RangeTracker& rangeTracker, Graph& graph)
        : m_RangeTracker(rangeTracker),
          m_Graph(graph)
{}

void DynamicQuantizationStrategy::SetRange(const IConnectableLayer* layer, unsigned int outputIdx, float min, float max)
{
    m_RangeTracker.SetRange(layer, outputIdx, min, max);
}

void DynamicQuantizationStrategy::ForwardParentParameters(const IConnectableLayer* layer)
{
    for (unsigned int i = 0; i < layer->GetNumInputSlots(); ++i)
    {
        const IOutputSlot *outputSlot = layer->GetInputSlot(i).GetConnection();
        LayerGuid previousLayerId = outputSlot->GetOwningLayerGuid();
        unsigned int ownerIndex = outputSlot->CalculateIndexOnOwner();
        const auto parentRange = m_RangeTracker.GetRange(previousLayerId, ownerIndex);
        SetRange(layer, i, parentRange.first, parentRange.second);
    }
}

void DynamicQuantizationStrategy::AddToCalibratedLayers(const IConnectableLayer* layer)
{
    m_LayersToCalibrate.push_back(layer);
}

void DynamicQuantizationStrategy::AddToNonCalibratedLayers(const IConnectableLayer* layer)
{
    m_LayersNotToCalibrate.push_back(layer);
}

void DynamicQuantizationStrategy::FinishStrategy()
{
    for (const IConnectableLayer* layer : m_LayersToCalibrate)
    {
        std::vector<DebugLayer*> newDebugLayers = InsertDebugLayerAfter(
            m_Graph, *PolymorphicDowncast<Layer*>(const_cast<IConnectableLayer*>(layer)));
        // record them so we can take them out again efficiently afterward
        m_DebugLayers.insert(std::end(m_DebugLayers), std::begin(newDebugLayers), std::end(newDebugLayers));
    }
}

void DynamicQuantizationStrategy::RemoveDebugLayers()
{
    for (DebugLayer* debugLayer : m_DebugLayers)
    {
        OutputSlot& proceedingOutputSlot = *debugLayer->GetInputSlot(0).GetConnectedOutputSlot();
        proceedingOutputSlot.Disconnect(debugLayer->GetInputSlot(0));

        for (InputSlot* succeedingInputSlot : debugLayer->GetOutputSlot(0).GetConnections())
        {
            debugLayer->GetOutputSlot(0).Disconnect(*succeedingInputSlot);
            proceedingOutputSlot.Connect(*succeedingInputSlot);
        }
        m_Graph.EraseLayer(debugLayer);
    }
    m_DebugLayers.clear();
}

void DynamicQuantizationStrategy::VisitNonCalibratedLayers() {
    RemoveDebugLayers();
    for (const IConnectableLayer* layer : m_LayersNotToCalibrate)
    {
        ForwardParentParameters(layer);
    }
}


void DynamicQuantizationStrategy::ExecuteStrategy(const armnn::IConnectableLayer* layer,
                                                  const BaseDescriptor& descriptor,
                                                  const std::vector<armnn::ConstTensor>& constants,
                                                  const char* name,
                                                  const armnn::LayerBindingId id)
{
    IgnoreUnused(name);
    IgnoreUnused(id);
    IgnoreUnused(descriptor);

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
            AddToCalibratedLayers(layer);
            break;
        }
        case armnn::LayerType::ArgMinMax :
        {
            AddToNonCalibratedLayers(layer);
            break;
        }
        case armnn::LayerType::BatchNormalization :
        {
            SetRange(layer, 0, -15.0f, 15.0f);
            AddToCalibratedLayers(layer);
            break;
        }
        case armnn::LayerType::Normalization:
        {
            SetRange(layer, 0, -15.0f, 15.0f);
            AddToCalibratedLayers(layer);
            break;
        }
        case armnn::LayerType::Convolution2d:
        {
            SetRange(layer, 0, -15.0f, 15.0f);
            AddToCalibratedLayers(layer);
            break;
        }
        case armnn::LayerType::DepthwiseConvolution2d:
        {
            SetRange(layer, 0, -15.0f, 15.0f);
            AddToCalibratedLayers(layer);
            break;
        }
        case armnn::LayerType::FullyConnected :
        {
            SetRange(layer, 0, -15.0f, 15.0f);
            AddToCalibratedLayers(layer);
            break;
        }
        case armnn::LayerType::Permute :
        {
            AddToNonCalibratedLayers(layer);
            break;
        }
        case armnn::LayerType::SpaceToBatchNd :
        {
            AddToNonCalibratedLayers(layer);
            break;
        }
        case armnn::LayerType::Pooling2d :
        {
            AddToNonCalibratedLayers(layer);
            break;
        }
        case armnn::LayerType::Softmax :
        {
            SetRange(layer, 0, 0.f, 1.f);
            AddToCalibratedLayers(layer);
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
            AddToCalibratedLayers(layer);
            break;
        }
        case armnn::LayerType::Reshape :
        {
            AddToNonCalibratedLayers(layer);
            break;
        }
        case armnn::LayerType::Splitter :
        {
            AddToNonCalibratedLayers(layer);
            break;
        }
        case armnn::LayerType::Resize :
        {
            AddToNonCalibratedLayers(layer);
            break;
        }
        case armnn::LayerType::StridedSlice :
        {
            AddToNonCalibratedLayers(layer);
            break;
        }
        case armnn::LayerType::BatchToSpaceNd :
        {
            AddToNonCalibratedLayers(layer);
            break;
        }
        case armnn::LayerType::Input :
        {
            SetRange(layer, 0, -0.0f, 0.0f);
            AddToCalibratedLayers(layer);
            break;
        }
        case armnn::LayerType::Output :
        {
            AddToNonCalibratedLayers(layer);
            m_OutputLayers.push_back(id);
            break;
        }
        default:
        {}
    }
}

const std::vector<LayerBindingId>& DynamicQuantizationStrategy::GetOutputLayers()
{
    return m_OutputLayers;
}

} //namespace armnn
