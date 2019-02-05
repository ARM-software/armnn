//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Network.hpp"
#include "QuantizerVisitor.hpp"
#include "StaticRangeVisitor.hpp"

#include <cmath>
#include <stdint.h>
#include <limits>

namespace armnn
{

namespace {

std::pair<int, float> ComputeQAsymmParams(int numBits, double min, double max)
{
    BOOST_ASSERT_MSG(min < max, "Min >= max will result in invalid quantization.");
    double highest = (1 << numBits)-1;

    min = std::min(0.0, min); // min <= 0.0
    max = std::max(0.0, max); // max >= 0.0

    // assumes quantization range [0-highest]
    double scale = (max-min) / highest;
    double offset = -min / scale;

    // clamp offset [0-highest]
    offset = std::max(0.0, std::min(highest, offset));

    return std::make_pair(static_cast<int>(std::round(offset)), static_cast<float>(scale));
}

} // namespace

QuantizerVisitor::QuantizerVisitor(armnn::StaticRangeVisitor* ranges)
: m_Ranges(ranges)
, m_QuantizedNetwork(INetwork::Create())
{
}

void QuantizerVisitor::SetQuantizedInputConnections(const IConnectableLayer *srcLayer,
                                                    IConnectableLayer *quantizedLayer)
{
    m_OldToNewGuidMap[srcLayer->GetGuid()] = quantizedLayer->GetGuid();

    for (unsigned int i=0; i < srcLayer->GetNumInputSlots(); i++)
    {
        const IInputSlot& srcInputSlot = srcLayer->GetInputSlot(i);
        const InputSlot* inputSlot = boost::polymorphic_downcast<const InputSlot*>(&srcInputSlot);
        const OutputSlot* outputSlot = inputSlot->GetConnectedOutputSlot();

        unsigned int slotIdx = outputSlot->CalculateIndexOnOwner();
        Layer& layerToFind = outputSlot->GetOwningLayer();

        auto found = m_OldToNewGuidMap.find(layerToFind.GetGuid());
        if (found != m_OldToNewGuidMap.end())
        {
            // Connect the slots in the quantized model
            IConnectableLayer* prevQuantizedLayer = m_GuidToLayerMap[found->second];
            IInputSlot& newInputSlot = quantizedLayer->GetInputSlot(i);
            IOutputSlot& newOutputSlot = prevQuantizedLayer->GetOutputSlot(slotIdx);
            newOutputSlot.Connect(newInputSlot);

            // Fetch the min/max ranges that were computed earlier
            auto range = m_Ranges->GetRange(layerToFind.GetGuid(), i);
            auto qParams = ComputeQAsymmParams(8, range.first, range.second);

            // Set the quantization params
            TensorInfo info(newOutputSlot.GetTensorInfo());
            info.SetDataType(DataType::QuantisedAsymm8);
            info.SetQuantizationOffset(qParams.first);
            info.SetQuantizationScale(qParams.second);
        }
        else
        {
            // error in graph traversal order
            BOOST_ASSERT_MSG(false, "Error in graph traversal");
        }
    }
}

void QuantizerVisitor::RecordLayer(IConnectableLayer* layer)
{
    m_GuidToLayerMap[layer->GetGuid()] = layer;
}

void QuantizerVisitor::VisitAdditionLayer(const IConnectableLayer *layer, const char *name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddAdditionLayer(name);
    RecordLayer(newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitInputLayer(const IConnectableLayer *layer, LayerBindingId id, const char *name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddInputLayer(id, name);
    RecordLayer(newLayer);
}

void QuantizerVisitor::VisitOutputLayer(const IConnectableLayer *layer, LayerBindingId id, const char *name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddOutputLayer(id, name);
    RecordLayer(newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

} //namespace armnn