//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "QuantizerStrategy.hpp"
#include "armnn/utility/PolymorphicDowncast.hpp"

namespace armnn
{

QuantizerStrategy::QuantizerStrategy(const RangeTracker& rangeTracker,
                                   const IQuantizationScheme* quantizationScheme,
                                   bool preserveType)
        : m_Ranges(rangeTracker)
        , m_QuantizedNetwork(INetwork::Create())
        , m_QuantizationScheme(quantizationScheme)
        , m_PreserveType(preserveType)
{
}

void QuantizerStrategy::SetQuantizedInputConnections(const IConnectableLayer* srcLayer,
                                                     IConnectableLayer* quantizedLayer)
{
    ARMNN_ASSERT(srcLayer);
    for (unsigned int i = 0; i < srcLayer->GetNumInputSlots(); i++)
    {
        const IInputSlot& srcInputSlot = srcLayer->GetInputSlot(i);
        const InputSlot* inputSlot = static_cast<const InputSlot*>(&srcInputSlot);
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
        TensorInfo info(outputSlot->GetTensorInfo());

        // Only try to set quantization params on tensors that can be quantized
        if (inputSlot->GetConnectedOutputSlot()->GetTensorInfo().GetDataType() != DataType::Boolean &&
            inputSlot->GetConnectedOutputSlot()->GetTensorInfo().GetDataType() != DataType::Signed32 &&
            inputSlot->GetConnectedOutputSlot()->GetTensorInfo().GetDataType() != DataType::Signed64)
        {
            // Fetch the min/max ranges that were computed earlier
            auto range = m_Ranges.GetRange(layerToFind.GetGuid(), slotIdx);
            OffsetScalePair qParams = m_QuantizationScheme->ComputeScheme(range.first, range.second);
            info.SetDataType(m_QuantizationScheme->GetDataType());
            info.SetQuantizationOffset(qParams.second);
            info.SetQuantizationScale(qParams.first);
        }
        newOutputSlot.SetTensorInfo(info);
    }
}

ConstTensor QuantizerStrategy::CreateQuantizedBias(const IConnectableLayer* srcLayer,
                                                   const ConstTensor& weights,
                                                   const Optional<ConstTensor>& biases,
                                                   std::vector<int32_t>& backing)
{
    ARMNN_ASSERT(srcLayer);
    const IInputSlot& srcInputSlot = srcLayer->GetInputSlot(0);
    auto inputSlot = static_cast<const InputSlot*>(&srcInputSlot);
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

void QuantizerStrategy::RecordLayer(const IConnectableLayer* srcLayer, IConnectableLayer* quantizedLayer)
{
    m_OriginalToQuantizedGuidMap.insert(std::make_pair(srcLayer->GetGuid(), quantizedLayer->GetGuid()));
    m_QuantizedGuidToLayerMap.insert(std::make_pair(quantizedLayer->GetGuid(), quantizedLayer));
}

void QuantizerStrategy::ExecuteStrategy(const armnn::IConnectableLayer *layer,
                                        const BaseDescriptor& descriptor,
                                        const std::vector<armnn::ConstTensor> &constants,
                                        const char *name,
                                        const armnn::LayerBindingId id)
{
    IgnoreUnused(id);

    IConnectableLayer* newLayer;

    switch (layer->GetType())
    {
        case armnn::LayerType::Addition :
        {
            newLayer = m_QuantizedNetwork->AddAdditionLayer(name);
            break;
        }
        case armnn::LayerType::Activation :
        {
            const ActivationDescriptor& activationDescriptor = static_cast<const ActivationDescriptor&>(descriptor);
            newLayer = m_QuantizedNetwork->AddActivationLayer(activationDescriptor, name);
            break;
        }
        case armnn::LayerType::ArgMinMax :
        {
            ArgMinMaxDescriptor argMinMaxDescriptor = static_cast<const ArgMinMaxDescriptor&>(descriptor);
            newLayer = m_QuantizedNetwork->AddArgMinMaxLayer(argMinMaxDescriptor, name);
            break;
        }
        case armnn::LayerType::BatchNormalization :
        {

            BatchNormalizationDescriptor batchNormalizationDescriptor =
                    static_cast<const BatchNormalizationDescriptor&>(descriptor);
            std::vector<uint8_t> meanBacking;
            ConstTensor qMean = CreateQuantizedConst(constants[0], meanBacking);

            std::vector<uint8_t> varianceBacking;
            ConstTensor qVariance = CreateQuantizedConst(constants[1], varianceBacking);

            std::vector<uint8_t> betaBacking;
            ConstTensor qBeta = CreateQuantizedConst(constants[2], betaBacking);

            std::vector<uint8_t> gammaBacking;
            ConstTensor qGamma = CreateQuantizedConst(constants[3], gammaBacking);

            newLayer = m_QuantizedNetwork->AddBatchNormalizationLayer(batchNormalizationDescriptor,
                                                                                         qMean,
                                                                                         qVariance,
                                                                                         qBeta,
                                                                                         qGamma,
                                                                                         name);
            break;
        }
        case armnn::LayerType::BatchToSpaceNd :
        {
            BatchToSpaceNdDescriptor batchToSpaceNdDescriptor =
                    static_cast<const BatchToSpaceNdDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddBatchToSpaceNdLayer(batchToSpaceNdDescriptor, name);
            break;
        }
        case armnn::LayerType::Comparison :
        {
            ComparisonDescriptor comparisonDescriptor =static_cast<const ComparisonDescriptor&>(descriptor);
            newLayer = m_QuantizedNetwork->AddComparisonLayer(comparisonDescriptor, name);
            break;
        }
        case armnn::LayerType::Concat :
        {
            OriginsDescriptor originsDescriptor = static_cast<const OriginsDescriptor&>(descriptor);
            newLayer = m_QuantizedNetwork->AddConcatLayer(originsDescriptor, name);
            break;
        }
        case armnn::LayerType::Constant :
        {
            std::vector<uint8_t> inputBacking;
            ConstTensor qInput = CreateQuantizedConst(constants[0], inputBacking);

            newLayer = m_QuantizedNetwork->AddConstantLayer(qInput, name);
            break;
        }
        case armnn::LayerType::Convolution2d :
        {
            const armnn::Optional<ConstTensor> biases = constants.size() == 1 ?
                    armnn::Optional<ConstTensor>{} :
                    armnn::Optional<ConstTensor>(constants[1]);

            std::vector<uint8_t> weightsBacking;
            ConstTensor qWeights = CreateQuantizedConst(constants[0], weightsBacking);
            Optional<ConstTensor> optionalQBiases;
            std::vector<int32_t> biasesBacking;

            if (biases.has_value())
            {
                ConstTensor qBiases = CreateQuantizedBias(layer, qWeights, biases, biasesBacking);
                optionalQBiases = Optional<ConstTensor>(qBiases);
            }
            Convolution2dDescriptor convolution2dDescriptor = static_cast<const Convolution2dDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddConvolution2dLayer(convolution2dDescriptor,
                                                                 qWeights,
                                                                 optionalQBiases,
                                                                 name);
            break;
        }
        case armnn::LayerType::DepthToSpace :
        {
            DepthToSpaceDescriptor depthToSpaceDescriptor = static_cast<const DepthToSpaceDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddDepthToSpaceLayer(depthToSpaceDescriptor, name);
            break;
        }
        case armnn::LayerType::DepthwiseConvolution2d :
        {
            DepthwiseConvolution2dDescriptor depthwiseConvolution2dDescriptor =
                    static_cast<const DepthwiseConvolution2dDescriptor&>(descriptor);

            const armnn::Optional<ConstTensor> biases = constants.size() == 1 ?
                                                        armnn::Optional<ConstTensor>{} :
                                                        armnn::Optional<ConstTensor>(constants[1]);

            std::vector<uint8_t> weightsBacking;
            ConstTensor qWeights = CreateQuantizedConst(constants[0], weightsBacking);
            Optional<ConstTensor> optionalQBiases;
            std::vector<int32_t> biasesBacking;

            if (biases.has_value())
            {
                ConstTensor qBiases = CreateQuantizedBias(layer, qWeights, biases, biasesBacking);
                optionalQBiases = Optional<ConstTensor>(qBiases);
            }

            newLayer = m_QuantizedNetwork->AddDepthwiseConvolution2dLayer(
                    depthwiseConvolution2dDescriptor,
                    qWeights,
                    optionalQBiases,
                    name);
            break;
        }
        case armnn::LayerType::ElementwiseUnary :
        {
            ElementwiseUnaryDescriptor elementwiseUnaryDescriptor =
                    static_cast<const ElementwiseUnaryDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddElementwiseUnaryLayer(elementwiseUnaryDescriptor, name);
            break;
        }
        case armnn::LayerType::Fill :
        {
            FillDescriptor fillDescriptor = static_cast<const FillDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddFillLayer(fillDescriptor, name);
            break;
        }
        case armnn::LayerType::FullyConnected :
        {
            FullyConnectedDescriptor fullyConnectedDescriptor =
                    static_cast<const FullyConnectedDescriptor&>(descriptor);

            const armnn::Optional<ConstTensor> biases = constants.size() == 1 ?
                                                        armnn::Optional<ConstTensor>{} :
                                                        armnn::Optional<ConstTensor>(constants[1]);

            std::vector<uint8_t> weightsBacking;
            ConstTensor qWeights = CreateQuantizedConst(constants[0], weightsBacking);
            Optional<ConstTensor> optionalQBiases;
            std::vector<int32_t> biasesBacking;

            if (biases.has_value())
            {
                ConstTensor qBiases = CreateQuantizedBias(layer, qWeights, biases, biasesBacking);
                optionalQBiases = Optional<ConstTensor>(qBiases);
            }

            newLayer = m_QuantizedNetwork->AddFullyConnectedLayer(fullyConnectedDescriptor,
                                                                                     qWeights,
                                                                                     optionalQBiases,
                                                                                     name);
            break;
        }
        case armnn::LayerType::Input :
        {
            const DataType dataType = layer->GetOutputSlot(0).GetTensorInfo().GetDataType();
            IConnectableLayer* inputLayer = m_QuantizedNetwork->AddInputLayer(id, name);

            if (m_PreserveType && (dataType == DataType::Float32 || dataType == DataType::Float16))
            {
                IConnectableLayer* quantizeLayer = m_QuantizedNetwork->AddQuantizeLayer();
                inputLayer->GetOutputSlot(0).Connect(quantizeLayer->GetInputSlot(0));
                inputLayer->GetOutputSlot(0).SetTensorInfo(layer->GetOutputSlot(0).GetTensorInfo());
                RecordLayer(layer, quantizeLayer);
                return;
            }
            else
            {
                RecordLayer(layer, inputLayer);
                return;
            }
        }
        case armnn::LayerType::InstanceNormalization :
        {
            InstanceNormalizationDescriptor instanceNormalizationDescriptor =
                    static_cast<const InstanceNormalizationDescriptor&>(descriptor);

            newLayer =
                    m_QuantizedNetwork->AddInstanceNormalizationLayer(instanceNormalizationDescriptor, name);
            break;
        }
        case armnn::LayerType::LogSoftmax :
        {
            LogSoftmaxDescriptor logSoftmaxDescriptor = static_cast<const LogSoftmaxDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddLogSoftmaxLayer(logSoftmaxDescriptor, name);
            break;
        }
        case armnn::LayerType::Mean :
        {
            MeanDescriptor meanDescriptor = static_cast<const MeanDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddMeanLayer(meanDescriptor, name);
            break;
        }
        case armnn::LayerType::Multiplication :
        {
            newLayer = m_QuantizedNetwork->AddMultiplicationLayer(name);
            break;
        }
        case armnn::LayerType::Normalization :
        {
            NormalizationDescriptor normalizationDescriptor = static_cast<const NormalizationDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddNormalizationLayer(normalizationDescriptor, name);
            break;
        }
        case armnn::LayerType::Output :
        {
            const TensorInfo& info = layer->GetInputSlot(0).GetConnection()->GetTensorInfo();
            const DataType& dataType = info.GetDataType();
            newLayer = m_QuantizedNetwork->AddOutputLayer(id, name);

            if (m_PreserveType  && (dataType == DataType::Float32 || dataType == DataType::Float16))
            {
                IConnectableLayer* dequantizeLayer = m_QuantizedNetwork->AddDequantizeLayer();
                RecordLayer(layer, dequantizeLayer);
                SetQuantizedInputConnections(layer, dequantizeLayer);
                dequantizeLayer->GetOutputSlot(0).Connect(newLayer->GetInputSlot(0));
                dequantizeLayer->GetOutputSlot(0).SetTensorInfo(info);
                return;
            }
            else
            {
                break;
            }
        }
        case armnn::LayerType::Pad :
        {
            PadDescriptor padDescriptor = static_cast<const PadDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddPadLayer(padDescriptor, name);
            break;
        }
        case armnn::LayerType::Permute :
        {
            PermuteDescriptor permuteDescriptor = static_cast<const PermuteDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddPermuteLayer(permuteDescriptor, name);
            break;
        }
        case armnn::LayerType::Pooling2d :
        {
            Pooling2dDescriptor pooling2dDescriptor = static_cast<const Pooling2dDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddPooling2dLayer(pooling2dDescriptor, name);
            break;
        }
        case armnn::LayerType::Prelu :
        {
            newLayer = m_QuantizedNetwork->AddPreluLayer(name);
            break;
        }
        case armnn::LayerType::Reshape :
        {
            ReshapeDescriptor reshapeDescriptor = static_cast<const ReshapeDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddReshapeLayer(reshapeDescriptor, name);
            break;
        }
        case armnn::LayerType::Resize :
        {

            ResizeBilinearDescriptor resizeBilinearDescriptor =
                    static_cast<const ResizeBilinearDescriptor&>(descriptor);

            ResizeDescriptor resizeDescriptor;
            resizeDescriptor.m_Method       = ResizeMethod::Bilinear;
            resizeDescriptor.m_TargetWidth  = resizeBilinearDescriptor.m_TargetWidth;
            resizeDescriptor.m_TargetHeight = resizeBilinearDescriptor.m_TargetHeight;
            resizeDescriptor.m_DataLayout   = resizeBilinearDescriptor.m_DataLayout;

            newLayer = m_QuantizedNetwork->AddResizeLayer(resizeDescriptor, name);
            break;
        }
        case armnn::LayerType::Slice :
        {
            SliceDescriptor sliceDescriptor = static_cast<const SliceDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddSliceLayer(sliceDescriptor, name);
            break;
        }
        case armnn::LayerType::Softmax :
        {
            SoftmaxDescriptor softmaxDescriptor = static_cast<const SoftmaxDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddSoftmaxLayer(softmaxDescriptor, name);
            break;
        }
        case armnn::LayerType::SpaceToBatchNd :
        {
            SpaceToBatchNdDescriptor spaceToBatchNdDescriptor =
                    static_cast<const SpaceToBatchNdDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddSpaceToBatchNdLayer(spaceToBatchNdDescriptor, name);
            break;
        }
        case armnn::LayerType::SpaceToDepth :
        {
            SpaceToDepthDescriptor spaceToDepthDescriptor = static_cast<const SpaceToDepthDescriptor&>(descriptor);
            newLayer = m_QuantizedNetwork->AddSpaceToDepthLayer(spaceToDepthDescriptor, name);
            break;
        }
        case armnn::LayerType::Splitter :
        {
            SplitterDescriptor splitterDescriptor = static_cast<const SplitterDescriptor&>(descriptor);
            newLayer = m_QuantizedNetwork->AddSplitterLayer(splitterDescriptor, name);
            break;
        }
        case armnn::LayerType::Stack :
        {
            StackDescriptor stackDescriptor = static_cast<const StackDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddStackLayer(stackDescriptor, name);
            break;
        }
        case armnn::LayerType::StridedSlice :
        {
            StridedSliceDescriptor stridedSliceDescriptor = static_cast<const StridedSliceDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddStridedSliceLayer(stridedSliceDescriptor, name);
            break;
        }
        case armnn::LayerType::Subtraction :
        {
            newLayer = m_QuantizedNetwork->AddSubtractionLayer( name);
            break;
        }
        case armnn::LayerType::TransposeConvolution2d :
        {

            const armnn::Optional<ConstTensor> biases = constants.size() == 1 ?
                                                        armnn::Optional<ConstTensor>{} :
                                                        armnn::Optional<ConstTensor>(constants[1]);
            // quantize weights
            std::vector<uint8_t> weightsBacking;
            ConstTensor qWeights = CreateQuantizedConst(constants[0], weightsBacking);

            // quantize biases
            std::vector<int32_t> biasesBacking;
            Optional<ConstTensor> optionalQBiases;
            if (biases.has_value())
            {
                ConstTensor qBiases = CreateQuantizedBias(layer, qWeights, biases, biasesBacking);
                optionalQBiases = Optional<ConstTensor>(qBiases);
            }

            TransposeConvolution2dDescriptor transposeConvolution2dDescriptor =
                    static_cast<const TransposeConvolution2dDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddTransposeConvolution2dLayer(transposeConvolution2dDescriptor,
                                                                          qWeights,
                                                                          optionalQBiases,
                                                                          name);
            break;
        }
        case armnn::LayerType::Transpose :
        {
            TransposeDescriptor transposeDescriptor = static_cast<const TransposeDescriptor&>(descriptor);

            newLayer = m_QuantizedNetwork->AddTransposeLayer(transposeDescriptor, name);
            break;
        }
        default:
        {
            throw UnimplementedException("Unimplemented layer encountered");
        }
    }
    RecordLayer(layer, newLayer);
    SetQuantizedInputConnections(layer, newLayer);
}

}

