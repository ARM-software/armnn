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

QuantizerVisitor::QuantizerVisitor(const RangeTracker& rangeTracker,
                                   const IQuantizationScheme* quantizationScheme,
                                   bool preserveType)
    : m_Ranges(rangeTracker)
    , m_QuantizedNetwork(INetwork::Create())
    , m_QuantizationScheme(quantizationScheme)
    , m_PreserveType(preserveType)
{
}

void QuantizerVisitor::SetQuantizedInputConnections(const IConnectableLayer* srcLayer,
                                                    IConnectableLayer* quantizedLayer)
{
    for (unsigned int i = 0; i < srcLayer->GetNumInputSlots(); i++)
    {
        const IInputSlot& srcInputSlot = srcLayer->GetInputSlot(i);
        const InputSlot* inputSlot = boost::polymorphic_downcast<const InputSlot*>(&srcInputSlot);
        const OutputSlot* outputSlot = inputSlot->GetConnectedOutputSlot();

        unsigned int slotIdx = outputSlot->CalculateIndexOnOwner();
        Layer& layerToFind = outputSlot->GetOwningLayer();

        auto found = m_OriginalToQuantizedGuidMap.find(layerToFind.GetGuid());
        if (found == m_OriginalToQuantizedGuidMap.end())
        {
            // Error in graph traversal order
            BOOST_ASSERT_MSG(false, "Error in graph traversal");
            return;
        }

        // Connect the slots in the quantized model
        IConnectableLayer* prevQuantizedLayer = m_QuantizedGuidToLayerMap[found->second];
        IInputSlot& newInputSlot = quantizedLayer->GetInputSlot(i);
        IOutputSlot& newOutputSlot = prevQuantizedLayer->GetOutputSlot(slotIdx);
        newOutputSlot.Connect(newInputSlot);

        // Fetch the min/max ranges that were computed earlier
        auto range = m_Ranges.GetRange(layerToFind.GetGuid(), slotIdx);
        OffsetScalePair qParams = m_QuantizationScheme->ComputeScheme(range.first, range.second);

        // Set the quantization params
        TensorInfo info(outputSlot->GetTensorInfo());
        info.SetDataType(m_QuantizationScheme->GetDataType());
        info.SetQuantizationOffset(qParams.second);
        info.SetQuantizationScale(qParams.first);
        newOutputSlot.SetTensorInfo(info);
    }
}

void QuantizerVisitor::RecordLayer(const IConnectableLayer* srcLayer, IConnectableLayer* quantizedLayer)
{
    m_OriginalToQuantizedGuidMap[srcLayer->GetGuid()] = quantizedLayer->GetGuid();
    m_QuantizedGuidToLayerMap[quantizedLayer->GetGuid()] = quantizedLayer;
}

void QuantizerVisitor::VisitAdditionLayer(const IConnectableLayer* layer, const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddAdditionLayer(name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitActivationLayer(const IConnectableLayer* layer,
                                            const ActivationDescriptor& activationDescriptor,
                                            const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddActivationLayer(activationDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitFullyConnectedLayer(const IConnectableLayer *layer,
                                                const FullyConnectedDescriptor& desc,
                                                const ConstTensor& weights,
                                                const Optional<ConstTensor>& biases,
                                                const char *name)
{
    std::vector<uint8_t> weightsBacking;
    ConstTensor qWeights = CreateQuantizedConst(weights, weightsBacking);

    IConnectableLayer* newLayer;
    if (biases.has_value())
    {
        std::vector<uint8_t> biasBacking;
        ConstTensor qBias = CreateQuantizedConst(biases.value(), biasBacking);
        newLayer = m_QuantizedNetwork->AddFullyConnectedLayer(desc, qWeights, qBias, name);
    }
    else
    {
        newLayer = m_QuantizedNetwork->AddFullyConnectedLayer(desc, qWeights, name);
    }

    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitInputLayer(const IConnectableLayer *layer, LayerBindingId id, const char *name)
{
    const DataType dataType = layer->GetOutputSlot(0).GetTensorInfo().GetDataType();
    IConnectableLayer* inputLayer = m_QuantizedNetwork->AddInputLayer(id, name);

    if (m_PreserveType && (dataType == DataType::Float32 || dataType == DataType::Float16))
    {
        IConnectableLayer* quantizeLayer = m_QuantizedNetwork->AddQuantizeLayer();
        inputLayer->GetOutputSlot(0).Connect(quantizeLayer->GetInputSlot(0));
        inputLayer->GetOutputSlot(0).SetTensorInfo(layer->GetOutputSlot(0).GetTensorInfo());
        RecordLayer(layer, quantizeLayer);
    }
    else
    {
        RecordLayer(layer, inputLayer);
    }
}

void QuantizerVisitor::VisitOutputLayer(const IConnectableLayer* layer, LayerBindingId id, const char* name)
{
    const TensorInfo& info = layer->GetInputSlot(0).GetConnection()->GetTensorInfo();
    const DataType& dataType = info.GetDataType();
    IConnectableLayer* outputLayer = m_QuantizedNetwork->AddOutputLayer(id, name);

    if (m_PreserveType  && (dataType == DataType::Float32 || dataType == DataType::Float16))
    {
        IConnectableLayer* dequantizeLayer = m_QuantizedNetwork->AddDequantizeLayer();
        RecordLayer(layer, dequantizeLayer);
        SetQuantizedInputConnections(layer, dequantizeLayer);
        dequantizeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
        dequantizeLayer->GetOutputSlot(0).SetTensorInfo(info);
    }
    else
    {
        RecordLayer(layer, outputLayer);
        SetQuantizedInputConnections(layer, outputLayer);
    }
}

void QuantizerVisitor::VisitBatchNormalizationLayer(const IConnectableLayer* layer,
                                                    const BatchNormalizationDescriptor& desc,
                                                    const ConstTensor& mean,
                                                    const ConstTensor& variance,
                                                    const ConstTensor& beta,
                                                    const ConstTensor& gamma,
                                                    const char* name)
{
    std::vector<uint8_t> meanBacking;
    ConstTensor qMean = CreateQuantizedConst(mean, meanBacking);

    std::vector<uint8_t> varianceBacking;
    ConstTensor qVariance = CreateQuantizedConst(variance, varianceBacking);

    std::vector<uint8_t> betaBacking;
    ConstTensor qBeta = CreateQuantizedConst(beta, betaBacking);

    std::vector<uint8_t> gammaBacking;
    ConstTensor qGamma = CreateQuantizedConst(gamma, gammaBacking);

    IConnectableLayer* newLayer = m_QuantizedNetwork->AddBatchNormalizationLayer(desc,
                                                                                 qMean,
                                                                                 qVariance,
                                                                                 qBeta,
                                                                                 qGamma,
                                                                                 name);

    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitConvolution2dLayer(const IConnectableLayer* layer,
                                               const Convolution2dDescriptor& convolution2dDescriptor,
                                               const ConstTensor& weights,
                                               const Optional<ConstTensor>& biases,
                                               const char* name)
{
    std::vector<uint8_t> weightsBacking;
    ConstTensor qWeights = CreateQuantizedConst(weights, weightsBacking);

    IConnectableLayer* newLayer;
    if (biases.has_value())
    {
        std::vector<uint8_t> biasesBacking;
        ConstTensor qBiases = CreateQuantizedConst(biases.value(), biasesBacking);

        newLayer = m_QuantizedNetwork->AddConvolution2dLayer(convolution2dDescriptor,
                                                             qWeights,
                                                             qBiases,
                                                             name);
    }
    else
    {
        newLayer = m_QuantizedNetwork->AddConvolution2dLayer(convolution2dDescriptor, qWeights, name);
    }

    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitDepthwiseConvolution2dLayer(const IConnectableLayer* layer,
                                                        const DepthwiseConvolution2dDescriptor& desc,
                                                        const ConstTensor& weights,
                                                        const Optional<ConstTensor>& biases,
                                                        const char* name)
{
    std::vector<uint8_t> weightsBacking;
    ConstTensor qWeights = CreateQuantizedConst(weights, weightsBacking);

    IConnectableLayer* newLayer;
    if (biases.has_value())
    {
        std::vector<uint8_t> biasesBacking;
        ConstTensor qBiases = CreateQuantizedConst(biases.value(), biasesBacking);

        newLayer = m_QuantizedNetwork->AddDepthwiseConvolution2dLayer(desc,
                                                                      qWeights,
                                                                      qBiases,
                                                                      name);
    }
    else
    {
        newLayer = m_QuantizedNetwork->AddDepthwiseConvolution2dLayer(desc, qWeights, name);
    }

    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}


void QuantizerVisitor::VisitPermuteLayer(const IConnectableLayer* layer,
                                         const PermuteDescriptor& permuteDescriptor,
                                         const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddPermuteLayer(permuteDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitSpaceToBatchNdLayer(const IConnectableLayer* layer,
                                                const SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                                const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddSpaceToBatchNdLayer(spaceToBatchNdDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitPooling2dLayer(const IConnectableLayer* layer,
                                           const Pooling2dDescriptor& pooling2dDescriptor,
                                           const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddPooling2dLayer(pooling2dDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitMergerLayer(const IConnectableLayer* layer,
                                        const OriginsDescriptor& mergerDescriptor,
                                        const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddMergerLayer(mergerDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitSoftmaxLayer(const IConnectableLayer* layer,
                                         const SoftmaxDescriptor& softmaxDescriptor,
                                         const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddSoftmaxLayer(softmaxDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitConstantLayer(const IConnectableLayer* layer,
                                          const ConstTensor& input,
                                          const char* name)
{
    std::vector<uint8_t> inputBacking;
    ConstTensor qInput = CreateQuantizedConst(input, inputBacking);

    IConnectableLayer* newLayer = m_QuantizedNetwork->AddConstantLayer(qInput, name);
    RecordLayer(layer, newLayer);
}

void QuantizerVisitor::VisitSplitterLayer(const IConnectableLayer* layer,
                                          const SplitterDescriptor& splitterDescriptor,
                                          const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddSplitterLayer(splitterDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitReshapeLayer(const IConnectableLayer* layer,
                                         const ReshapeDescriptor& reshapeDescriptor,
                                         const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddReshapeLayer(reshapeDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitResizeBilinearLayer(const IConnectableLayer* layer,
                                                const ResizeBilinearDescriptor& resizeDesc,
                                                const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddResizeBilinearLayer(resizeDesc, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitStridedSliceLayer(const IConnectableLayer* layer,
                                              const StridedSliceDescriptor& stridedSliceDescriptor,
                                              const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddStridedSliceLayer(stridedSliceDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitBatchToSpaceNdLayer(const IConnectableLayer* layer,
                                                const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                                const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddBatchToSpaceNdLayer(batchToSpaceNdDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

} //namespace armnn
