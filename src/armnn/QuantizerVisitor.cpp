//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Network.hpp"
#include "NetworkQuantizerUtils.hpp"
#include "QuantizerVisitor.hpp"
#include "StaticRangeVisitor.hpp"

#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

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
    ARMNN_ASSERT(srcLayer);
    for (unsigned int i = 0; i < srcLayer->GetNumInputSlots(); i++)
    {
        const IInputSlot& srcInputSlot = srcLayer->GetInputSlot(i);
        const InputSlot* inputSlot = PolymorphicDowncast<const InputSlot*>(&srcInputSlot);
        ARMNN_ASSERT(inputSlot);
        const OutputSlot* outputSlot = inputSlot->GetConnectedOutputSlot();

        ARMNN_ASSERT(outputSlot);
        unsigned int slotIdx = outputSlot->CalculateIndexOnOwner();
        Layer& layerToFind = outputSlot->GetOwningLayer();

        auto found = m_OriginalToQuantizedGuidMap.find(layerToFind.GetGuid());
        if (found == m_OriginalToQuantizedGuidMap.end())
        {
            // Error in graph traversal order
            ARMNN_ASSERT_MSG(false, "Error in graph traversal");
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

ConstTensor QuantizerVisitor::CreateQuantizedBias(const IConnectableLayer* srcLayer,
                                                  const ConstTensor& weights,
                                                  const Optional<ConstTensor>& biases,
                                                  std::vector<int32_t>& backing)
{
    ARMNN_ASSERT(srcLayer);
    const IInputSlot& srcInputSlot = srcLayer->GetInputSlot(0);
    auto inputSlot = PolymorphicDowncast<const InputSlot*>(&srcInputSlot);
    ARMNN_ASSERT(inputSlot);
    const OutputSlot* outputSlot = inputSlot->GetConnectedOutputSlot();

    ARMNN_ASSERT(outputSlot);
    unsigned int slotIdx = outputSlot->CalculateIndexOnOwner();
    Layer& layerToFind = outputSlot->GetOwningLayer();

    auto found = m_OriginalToQuantizedGuidMap.find(layerToFind.GetGuid());
    if (found == m_OriginalToQuantizedGuidMap.end())
    {
        // Error in graph traversal order
        ARMNN_ASSERT_MSG(false, "Error in graph traversal");
        return biases.value();
    }

    // Fetch the min/max ranges that were computed earlier
    auto range = m_Ranges.GetRange(layerToFind.GetGuid(), slotIdx);
    OffsetScalePair qParams = m_QuantizationScheme->ComputeScheme(range.first, range.second);

    // Get the quantization scale based on input and weight scale
    float scale = qParams.first * weights.GetInfo().GetQuantizationScale();

    // Set up quantized bias tensor info and allocate space
    TensorInfo qInfo(biases.value().GetInfo().GetShape(), DataType::Signed32, scale, 0);
    backing.resize(biases.value().GetInfo().GetNumElements());

    // Convert values to int32
    for (size_t i = 0; i < backing.size(); ++i)
    {
        float fp32Value = static_cast<const float*>(biases.value().GetMemoryArea())[i];
        backing[i] = armnn::numeric_cast<int32_t>(fp32Value * ( 1 / scale ));
    }

    return ConstTensor(qInfo, backing);
}

void QuantizerVisitor::RecordLayer(const IConnectableLayer* srcLayer, IConnectableLayer* quantizedLayer)
{
    m_OriginalToQuantizedGuidMap.insert(std::make_pair(srcLayer->GetGuid(), quantizedLayer->GetGuid()));
    m_QuantizedGuidToLayerMap.insert(std::make_pair(quantizedLayer->GetGuid(), quantizedLayer));
}

void QuantizerVisitor::VisitAbsLayer(const IConnectableLayer* layer, const char* name)
{
    VisitElementwiseUnaryLayer(layer, ElementwiseUnaryDescriptor(UnaryOperation::Abs), name);
}

void QuantizerVisitor::VisitActivationLayer(const IConnectableLayer* layer,
                                            const ActivationDescriptor& activationDescriptor,
                                            const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddActivationLayer(activationDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitAdditionLayer(const IConnectableLayer* layer, const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddAdditionLayer(name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitArgMinMaxLayer(const IConnectableLayer* layer,
                                           const ArgMinMaxDescriptor& argMinMaxDescriptor,
                                           const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddArgMinMaxLayer(argMinMaxDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
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

void QuantizerVisitor::VisitBatchToSpaceNdLayer(const IConnectableLayer* layer,
                                                const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                                const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddBatchToSpaceNdLayer(batchToSpaceNdDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitComparisonLayer(const IConnectableLayer* layer,
                                            const ComparisonDescriptor& comparisonDescriptor,
                                            const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddComparisonLayer(comparisonDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitConcatLayer(const IConnectableLayer* layer,
                                        const OriginsDescriptor& originsDescriptor,
                                        const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddConcatLayer(originsDescriptor, name);
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

void QuantizerVisitor::VisitConvolution2dLayer(const IConnectableLayer* layer,
                                               const Convolution2dDescriptor& convolution2dDescriptor,
                                               const ConstTensor& weights,
                                               const Optional<ConstTensor>& biases,
                                               const char* name)
{
    std::vector<uint8_t> weightsBacking;
    ConstTensor qWeights = CreateQuantizedConst(weights, weightsBacking);
    Optional<ConstTensor> optionalQBiases;
    std::vector<int32_t> biasesBacking;

    if (biases.has_value())
    {
        ConstTensor qBiases = CreateQuantizedBias(layer, qWeights, biases, biasesBacking);
        optionalQBiases = Optional<ConstTensor>(qBiases);
    }

    IConnectableLayer* newLayer = m_QuantizedNetwork->AddConvolution2dLayer(convolution2dDescriptor,
                                                                            qWeights,
                                                                            optionalQBiases,
                                                                            name);

    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitDepthToSpaceLayer(const IConnectableLayer* layer,
                                              const DepthToSpaceDescriptor& descriptor,
                                              const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddDepthToSpaceLayer(descriptor, name);
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
    Optional<ConstTensor> optionalQBiases;
    std::vector<int32_t> biasesBacking;

    if (biases.has_value())
    {
        ConstTensor qBiases = CreateQuantizedBias(layer, qWeights, biases, biasesBacking);
        optionalQBiases = Optional<ConstTensor>(qBiases);
    }

    IConnectableLayer* newLayer = m_QuantizedNetwork->AddDepthwiseConvolution2dLayer(desc,
                                                                                     qWeights,
                                                                                     optionalQBiases,
                                                                                     name);

    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitElementwiseUnaryLayer(const IConnectableLayer* layer,
                                                  const ElementwiseUnaryDescriptor& elementwiseUnaryDescriptor,
                                                  const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddElementwiseUnaryLayer(elementwiseUnaryDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitFillLayer(const IConnectableLayer* layer,
                                      const FillDescriptor& desc,
                                      const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddFillLayer(desc, name);
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
    Optional<ConstTensor> optionalQBiases;
    std::vector<int32_t> biasesBacking;

    if (biases.has_value())
    {
        ConstTensor qBiases = CreateQuantizedBias(layer, qWeights, biases, biasesBacking);
        optionalQBiases = Optional<ConstTensor>(qBiases);
    }

    IConnectableLayer* newLayer = m_QuantizedNetwork->AddFullyConnectedLayer(desc,
                                                                             qWeights,
                                                                             optionalQBiases,
                                                                             name);

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

void QuantizerVisitor::VisitInstanceNormalizationLayer(const IConnectableLayer* layer,
                                                       const InstanceNormalizationDescriptor& descriptor,
                                                       const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddInstanceNormalizationLayer(descriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitLogSoftmaxLayer(const IConnectableLayer* layer,
                                            const LogSoftmaxDescriptor& logSoftmaxDescriptor,
                                            const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddLogSoftmaxLayer(logSoftmaxDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitMeanLayer(const IConnectableLayer* layer,
                                      const MeanDescriptor& meanDescriptor,
                                      const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddMeanLayer(meanDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitMultiplicationLayer(const IConnectableLayer* layer,
                                                const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddMultiplicationLayer(name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitNormalizationLayer(const armnn::IConnectableLayer* layer,
                                               const armnn::NormalizationDescriptor& normalizationDescriptor,
                                               const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddNormalizationLayer(normalizationDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
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

void QuantizerVisitor::VisitPadLayer(const IConnectableLayer* layer,
                                     const PadDescriptor& padDescriptor,
                                     const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddPadLayer(padDescriptor, name);
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

void QuantizerVisitor::VisitPooling2dLayer(const IConnectableLayer* layer,
                                           const Pooling2dDescriptor& pooling2dDescriptor,
                                           const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddPooling2dLayer(pooling2dDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitPreluLayer(const IConnectableLayer* layer,
                                       const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddPreluLayer(name);
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
                                                const ResizeBilinearDescriptor& resizeBilinearDescriptor,
                                                const char* name)
{
    ResizeDescriptor resizeDescriptor;
    resizeDescriptor.m_Method       = ResizeMethod::Bilinear;
    resizeDescriptor.m_TargetWidth  = resizeBilinearDescriptor.m_TargetWidth;
    resizeDescriptor.m_TargetHeight = resizeBilinearDescriptor.m_TargetHeight;
    resizeDescriptor.m_DataLayout   = resizeBilinearDescriptor.m_DataLayout;

    VisitResizeLayer(layer, resizeDescriptor, name);
}

void QuantizerVisitor::VisitResizeLayer(const IConnectableLayer* layer,
                                        const ResizeDescriptor& resizeDescriptor,
                                        const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddResizeLayer(resizeDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitRsqrtLayer(const IConnectableLayer* layer, const char* name)
{
    VisitElementwiseUnaryLayer(layer, ElementwiseUnaryDescriptor(UnaryOperation::Rsqrt), name);
}

void QuantizerVisitor::VisitSliceLayer(const IConnectableLayer* layer,
                                       const SliceDescriptor& sliceDescriptor,
                                       const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddSliceLayer(sliceDescriptor, name);
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

void QuantizerVisitor::VisitSpaceToBatchNdLayer(const IConnectableLayer* layer,
                                                const SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                                const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddSpaceToBatchNdLayer(spaceToBatchNdDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitSpaceToDepthLayer(const IConnectableLayer* layer,
                                              const SpaceToDepthDescriptor& spaceToDepthDescriptor,
                                              const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddSpaceToDepthLayer(spaceToDepthDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitSplitterLayer(const IConnectableLayer* layer,
                                          const SplitterDescriptor& splitterDescriptor,
                                          const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddSplitterLayer(splitterDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitStackLayer(const IConnectableLayer* layer,
                                       const StackDescriptor& stackDescriptor,
                                       const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddStackLayer(stackDescriptor, name);
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

void QuantizerVisitor::VisitSubtractionLayer(const IConnectableLayer* layer,
                                                const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddSubtractionLayer(name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitTransposeConvolution2dLayer(const IConnectableLayer* layer,
                                                        const TransposeConvolution2dDescriptor& descriptor,
                                                        const ConstTensor& weights,
                                                        const Optional<ConstTensor>& biases,
                                                        const char* name)
{
    // quantize weights
    std::vector<uint8_t> weightsBacking;
    ConstTensor qWeights = CreateQuantizedConst(weights, weightsBacking);

    // quantize biases
    std::vector<int32_t> biasesBacking;
    Optional<ConstTensor> optionalQBiases;
    if (biases.has_value())
    {
        ConstTensor qBiases = CreateQuantizedBias(layer, qWeights, biases, biasesBacking);
        optionalQBiases = Optional<ConstTensor>(qBiases);
    }

    IConnectableLayer* newLayer = m_QuantizedNetwork->AddTransposeConvolution2dLayer(descriptor,
                                                                                     qWeights,
                                                                                     optionalQBiases,
                                                                                     name);

    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

void QuantizerVisitor::VisitTransposeLayer(const IConnectableLayer* layer,
                                           const TransposeDescriptor& transposeDescriptor,
                                           const char* name)
{
    IConnectableLayer* newLayer = m_QuantizedNetwork->AddTransposeLayer(transposeDescriptor, name);
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

} //namespace armnn
