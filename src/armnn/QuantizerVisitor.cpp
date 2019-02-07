//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Network.hpp"
#include "QuantizerVisitor.hpp"
#include "StaticRangeVisitor.hpp"

#include "armnn/TypesUtils.hpp"

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

template<typename srcType>
void Quantize(const srcType* src, uint8_t* dst, size_t numElements, float &scale, int &offset)
{
    BOOST_ASSERT(src);
    BOOST_ASSERT(dst);

    float min = std::numeric_limits<srcType>::max();
    float max = std::numeric_limits<srcType>::lowest();
    for (size_t i = 0; i < numElements; ++i)
    {
        min = std::min(min, src[i]);
        max = std::max(max, src[i]);
    }

    auto qParams = ComputeQAsymmParams(8, min, max);
    offset = qParams.first;
    scale = qParams.second;
    for (size_t i = 0; i < numElements; ++i)
    {
        dst[i] = armnn::Quantize<uint8_t>(src[i], scale, offset);
    }
}

ConstTensor CreateQuantizedConst(const ConstTensor& tensor, std::vector<uint8_t> &backing)
{
    float scale = 0.0f;
    int offset = 0;
    // Reserve the backing memory
    backing.resize(tensor.GetInfo().GetNumElements());

    DataType type = tensor.GetInfo().GetDataType();
    switch(type)
    {
        case DataType::Float32:
        {
            Quantize(static_cast<const float*>( tensor.GetMemoryArea()),
                     backing.data(),
                     backing.size(),
                     scale,
                     offset);
        }
            break;
        default:
            BOOST_ASSERT_MSG(false, "Can't quantize unsupported data type");
    }

    TensorInfo qInfo(tensor.GetInfo().GetShape(), DataType::QuantisedAsymm8, scale, offset);
    return ConstTensor(qInfo, backing);
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
            newOutputSlot.SetTensorInfo(info);
        }
        else
        {
            // error in graph traversal order
            BOOST_ASSERT_MSG(false, "Error in graph traversal");
        }
    }
}

void QuantizerVisitor::RecordLayer(const IConnectableLayer* srcLayer, IConnectableLayer* quantizedLayer)
{
    m_OldToNewGuidMap[srcLayer->GetGuid()] = quantizedLayer->GetGuid();
    m_GuidToLayerMap[quantizedLayer->GetGuid()] = quantizedLayer;
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