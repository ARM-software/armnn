//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Network.hpp"
#include "QuantizerVisitor.hpp"
#include "StaticRangeVisitor.hpp"
#include "NetworkQuantizerUtils.hpp"

namespace armnn
{

QuantizerVisitor::QuantizerVisitor(const StaticRangeVisitor *staticRangeVisitor)
    : m_StaticRangeVisitor(staticRangeVisitor)
    , m_QuantizedNetwork(INetwork::Create())
{
    BOOST_ASSERT(m_StaticRangeVisitor);
}

void QuantizerVisitor::SetQuantizedInputConnections(const IConnectableLayer *srcLayer,
                                                    IConnectableLayer *quantizedLayer)
{
    for (unsigned int i=0; i < srcLayer->GetNumInputSlots(); i++)
    {
        const IInputSlot& srcInputSlot = srcLayer->GetInputSlot(i);
        const InputSlot* inputSlot = boost::polymorphic_downcast<const InputSlot*>(&srcInputSlot);
        const OutputSlot* outputSlot = inputSlot->GetConnectedOutputSlot();

        unsigned int slotIdx = outputSlot->CalculateIndexOnOwner();
        Layer& layerToFind = outputSlot->GetOwningLayer();

        auto found = m_OriginalToQuantizedGuidMap.find(layerToFind.GetGuid());
        if (found != m_OriginalToQuantizedGuidMap.end())
        {
            // Connect the slots in the quantized model
            IConnectableLayer* prevQuantizedLayer = m_QuantizedGuidToLayerMap[found->second];
            IInputSlot& newInputSlot = quantizedLayer->GetInputSlot(i);
            IOutputSlot& newOutputSlot = prevQuantizedLayer->GetOutputSlot(slotIdx);
            newOutputSlot.Connect(newInputSlot);

            // Fetch the min/max ranges that were computed earlier
            auto range = m_StaticRangeVisitor->GetRange(layerToFind.GetGuid(), i);
            auto qParams = ComputeQAsymmParams(8, range.first, range.second);

            // Set the quantization params
            TensorInfo info(newOutputSlot.GetTensorInfo());
            info.SetDataType(DataType::QuantisedAsymm8);
            info.SetQuantizationOffset(qParams.first);
            info.SetQuantizationScale(qParams.second);
            newOutputSlot.SetTensorInfo(info);
        }
        else
        {
            // Error in graph traversal order
            BOOST_ASSERT_MSG(false, "Error in graph traversal");
        }
    }
}

void QuantizerVisitor::RecordLayer(const IConnectableLayer* srcLayer, IConnectableLayer* quantizedLayer)
{
    m_OriginalToQuantizedGuidMap[srcLayer->GetGuid()] = quantizedLayer->GetGuid();
    m_QuantizedGuidToLayerMap[quantizedLayer->GetGuid()] = quantizedLayer;
}

void QuantizerVisitor::VisitAdditionLayer(const IConnectableLayer *layer, const char *name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddAdditionLayer(name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitActivationLayer(const IConnectableLayer *layer,
                                            const ActivationDescriptor& activationDescriptor,
                                            const char *name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddActivationLayer(activationDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitInputLayer(const IConnectableLayer *layer, LayerBindingId id, const char *name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddInputLayer(id, name);
    RecordLayer(layer, newLayer);
}

void QuantizerVisitor::VisitOutputLayer(const IConnectableLayer *layer, LayerBindingId id, const char *name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddOutputLayer(id, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitBatchNormalizationLayer(const IConnectableLayer *layer,
                                                    const BatchNormalizationDescriptor& desc,
                                                    const ConstTensor& mean,
                                                    const ConstTensor& variance,
                                                    const ConstTensor& beta,
                                                    const ConstTensor& gamma,
                                                    const char *name)
{
    std::vector<uint8_t> meanBacking;
    ConstTensor qMean = CreateQuantizedConst(mean, meanBacking);

    std::vector<uint8_t> varianceBacking;
    ConstTensor qVariance = CreateQuantizedConst(variance, varianceBacking);

    std::vector<uint8_t> betaBacking;
    ConstTensor qBeta = CreateQuantizedConst(beta, betaBacking);

    std::vector<uint8_t> gammaBacking;
    ConstTensor qGamma = CreateQuantizedConst(variance, gammaBacking);

    IConnectableLayer* newLayer = m_QuantizedNetwork->AddBatchNormalizationLayer(desc,
                                                                                 qMean,
                                                                                 qVariance,
                                                                                 qBeta,
                                                                                 qGamma,
                                                                                 name);

    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

} //namespace armnn
