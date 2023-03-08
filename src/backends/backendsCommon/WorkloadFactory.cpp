//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <Layer.hpp>
#include <LayersFwd.hpp>

#include <armnn/Types.hpp>
#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/ILayerSupport.hpp>
#include <armnn/BackendHelper.hpp>
#include <armnn/BackendRegistry.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/utility/TransformIterator.hpp>

#include <armnn/backends/WorkloadFactory.hpp>

#include <sstream>

namespace armnn
{

namespace
{
using LayerList = std::list<Layer*>;
using Iterator = LayerList::const_iterator; // Const so pointers in the list can't be modified externally.

const TensorInfo OverrideDataType(const TensorInfo& info, Optional<DataType> type)
{
    if (!type)
    {
        return info;
    }

    return TensorInfo(info.GetShape(),
                      type.value(),
                      info.GetQuantizationScale(),
                      info.GetQuantizationOffset(),
                      info.IsConstant());
}

} // anonymous namespace

inline armnn::Optional<armnn::DataType> GetBiasTypeFromWeightsType(armnn::Optional<armnn::DataType> weightsType)
{
    if (!weightsType)
    {
        return weightsType;
    }

    switch(weightsType.value())
    {
        case armnn::DataType::BFloat16:
        case armnn::DataType::Float16:
        case armnn::DataType::Float32:
            return weightsType;
        case armnn::DataType::QAsymmS8:
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::QSymmS8:
        case armnn::DataType::QSymmS16:
            return armnn::DataType::Signed32;
        default:
            ARMNN_ASSERT_MSG(false, "GetBiasTypeFromWeightsType(): Unsupported data type.");
    }
    return armnn::EmptyOptional();
}


bool IWorkloadFactory::IsLayerConfigurationSupported(const BackendId& backendId,
                                                     const IConnectableLayer& connectableLayer,
                                                     Optional<DataType> dataType,
                                                     std::string& outReasonIfUnsupported,
                                                     const ModelOptions& modelOptions)
{
    Optional<std::string&> reason = outReasonIfUnsupported;
    bool result;
    const Layer& layer = *(PolymorphicDowncast<const Layer*>(&connectableLayer));

    auto const& backendRegistry = BackendRegistryInstance();
    if (!backendRegistry.IsBackendRegistered(backendId))
    {
        std::stringstream ss;
        ss << connectableLayer.GetName() << " is not supported on " << backendId
           << " because this backend is not registered.";

        outReasonIfUnsupported = ss.str();
        return false;
    }

    auto backendFactory = backendRegistry.GetFactory(backendId);
    auto backendObject = backendFactory();
    auto layerSupport = backendObject->GetLayerSupport(modelOptions);
    auto layerSupportObject = LayerSupportHandle(layerSupport, backendId);

    switch(layer.GetType())
    {
        case LayerType::Activation:
        {
            auto cLayer = PolymorphicDowncast<const ActivationLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsActivationSupported(
                                           OverrideDataType(input, dataType),
                                           OverrideDataType(output, dataType),
                                           cLayer->GetParameters(),
                                           reason);
            break;
        }
        case LayerType::Addition:
        {
            ARMNN_NO_DEPRECATE_WARN_BEGIN
            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsAdditionSupported(
                                        OverrideDataType(input0, dataType),
                                        OverrideDataType(input1, dataType),
                                        OverrideDataType(output, dataType),
                                        reason);
            ARMNN_NO_DEPRECATE_WARN_END
            break;
        }
        case LayerType::ArgMinMax:
        {
            auto cLayer = PolymorphicDowncast<const ArgMinMaxLayer*>(&layer);
            const ArgMinMaxDescriptor& descriptor = cLayer->GetParameters();

            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsArgMinMaxSupported(
                    OverrideDataType(input, dataType),
                    OverrideDataType(output, DataType::Signed32),
                    descriptor,
                    reason);
            break;
        }
        case LayerType::BatchMatMul:
        {
            auto cLayer = PolymorphicDowncast<const BatchMatMulLayer*>(&layer);
            const BatchMatMulDescriptor& descriptor = cLayer->GetParameters();

            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsBatchMatMulSupported(
                            OverrideDataType(input0, dataType),
                            OverrideDataType(input1, dataType),
                            OverrideDataType(output, dataType),
                            descriptor,
                            reason);
            break;
        }
        case LayerType::BatchNormalization:
        {
            auto cLayer = PolymorphicDowncast<const BatchNormalizationLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            const TensorInfo& mean = cLayer->m_Mean->GetTensorInfo();
            const TensorInfo& var = cLayer->m_Variance->GetTensorInfo();
            const TensorInfo& beta = cLayer->m_Beta->GetTensorInfo();
            const TensorInfo& gamma = cLayer->m_Gamma->GetTensorInfo();
            result = layerSupportObject.IsBatchNormalizationSupported(
                                                   OverrideDataType(input, dataType),
                                                   OverrideDataType(output, dataType),
                                                   OverrideDataType(mean, dataType),
                                                   OverrideDataType(var, dataType),
                                                   OverrideDataType(beta, dataType),
                                                   OverrideDataType(gamma, dataType),
                                                   cLayer->GetParameters(),
                                                   reason);
            break;
        }
        case LayerType::BatchToSpaceNd:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            auto cLayer = PolymorphicDowncast<const BatchToSpaceNdLayer*>(&layer);

            result = layerSupportObject.IsBatchToSpaceNdSupported(OverrideDataType(input, dataType),
                                                                  OverrideDataType(output, dataType),
                                                                  cLayer->GetParameters(),
                                                                  reason);
            break;
        }
        case LayerType::Cast:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsCastSupported(OverrideDataType(input, dataType),
                                                        OverrideDataType(output, dataType),
                                                        reason);
            break;
        }
        case LayerType::ChannelShuffle:
        {
            auto cLayer = PolymorphicDowncast<const ChannelShuffleLayer*>(&layer);

            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();

            const ChannelShuffleDescriptor descriptor = cLayer->GetParameters();

            result = layerSupportObject.IsChannelShuffleSupported(OverrideDataType(input, dataType),
                                                                  OverrideDataType(output, dataType),
                                                                  descriptor,
                                                                  reason);
            break;
        }
        case LayerType::Comparison:
        {
            auto cLayer = PolymorphicDowncast<const ComparisonLayer*>(&layer);

            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsComparisonSupported(OverrideDataType(input0, dataType),
                                                              OverrideDataType(input1, dataType),
                                                              OverrideDataType(output, DataType::Boolean),
                                                              cLayer->GetParameters(),
                                                              reason);
            break;
        }
        case LayerType::Constant:
        {
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsConstantSupported(OverrideDataType(output, dataType), reason);
            break;
        }
        case LayerType::ConvertFp16ToFp32:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsConvertFp16ToFp32Supported(input, output, reason);
            break;
        }
        case LayerType::ConvertFp32ToFp16:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsConvertFp32ToFp16Supported(input, output, reason);
            break;
        }
        case LayerType::Convolution2d:
        {
            auto cLayer = PolymorphicDowncast<const Convolution2dLayer*>(&layer);

            const TensorInfo input  = OverrideDataType(layer.GetInputSlot(0).GetConnection()->GetTensorInfo(),
                                                       dataType);
            const TensorInfo output = OverrideDataType(layer.GetOutputSlot(0).GetTensorInfo(), dataType);
            ARMNN_ASSERT_MSG(layer.GetInputSlot(1).GetConnection(),
                             "Convolution2dLayer: Weights should be connected as a Constant Layer.");
            const TensorInfo weights = OverrideDataType(layer.GetInputSlot(1).GetConnection()->GetTensorInfo(),
                                                        dataType);

            const Convolution2dDescriptor& descriptor = cLayer->GetParameters();

            // Construct optional biases object based on the value of m_BiasEnabled
            Optional<TensorInfo> biases;
            if (descriptor.m_BiasEnabled)
            {
                ARMNN_ASSERT_MSG(layer.GetInputSlot(2).GetConnection(),
                                 "Convolution2dLayer: Bias should be connected as a Constant Layer.");
                biases = OverrideDataType(layer.GetInputSlot(2).GetConnection()->GetTensorInfo(),
                                          GetBiasTypeFromWeightsType(dataType));
            }

            result = layerSupportObject.IsConvolution2dSupported(
                                              input,
                                              output,
                                              descriptor,
                                              weights,
                                              biases,
                                              reason);
            break;
        }
        case LayerType::Convolution3d:
        {
            auto cLayer = PolymorphicDowncast<const Convolution3dLayer*>(&layer);

            const TensorInfo input  = OverrideDataType(layer.GetInputSlot(0).GetConnection()->GetTensorInfo(),
                                                       dataType);
            const TensorInfo output = OverrideDataType(layer.GetOutputSlot(0).GetTensorInfo(), dataType);

            ARMNN_ASSERT_MSG(layer.GetInputSlot(1).GetConnection(),
                             "Convolution3dLayer: Weights should be connected as a Constant Layer.");
            const TensorInfo weights = OverrideDataType(layer.GetInputSlot(1).GetConnection()->GetTensorInfo(),
                                                        dataType);

            const Convolution3dDescriptor& descriptor = cLayer->GetParameters();

            // Construct optional biases object based on the value of m_BiasEnabled
            Optional<TensorInfo> biases;
            if (descriptor.m_BiasEnabled)
            {
                biases = OverrideDataType(layer.GetInputSlot(2).GetConnection()->GetTensorInfo(),
                                          GetBiasTypeFromWeightsType(dataType));
            }

            result = layerSupportObject.IsConvolution3dSupported(
                                              input,
                                              output,
                                              descriptor,
                                              weights,
                                              biases,
                                              reason);
            break;
        }
        case LayerType::Debug:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsDebugSupported(OverrideDataType(input, dataType),
                                                          OverrideDataType(output, dataType),
                                                          reason);
            break;
        }
        case LayerType::DepthToSpace:
        {
            auto cLayer = PolymorphicDowncast<const DepthToSpaceLayer*>(&layer);

            const TensorInfo& input  = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsDepthToSpaceSupported(OverrideDataType(input, dataType),
                                                                 OverrideDataType(output, dataType),
                                                                 cLayer->GetParameters(),
                                                                 reason);
            break;
        }
        case LayerType::DepthwiseConvolution2d:
        {
            auto cLayer = PolymorphicDowncast<const DepthwiseConvolution2dLayer*>(&layer);
            const TensorInfo& input   = OverrideDataType(layer.GetInputSlot(0).GetConnection()->GetTensorInfo(),
                                                         dataType);
            const TensorInfo& output  = OverrideDataType(layer.GetOutputSlot(0).GetTensorInfo(), dataType);
            const TensorInfo& weights = OverrideDataType(layer.GetInputSlot(1).GetConnection()->GetTensorInfo(),
                                                         dataType);

            ARMNN_ASSERT(cLayer->GetInputSlot(1).GetConnection() != nullptr);

            const DepthwiseConvolution2dDescriptor& descriptor = cLayer->GetParameters();

            // Construct optional biases object based on the value of m_BiasEnabled
            Optional<TensorInfo> biases;
            if (descriptor.m_BiasEnabled)
            {
                biases = OverrideDataType(cLayer->GetInputSlot(2).GetConnection()->GetTensorInfo(),
                                          GetBiasTypeFromWeightsType(dataType));
            }

            result = layerSupportObject.IsDepthwiseConvolutionSupported(input,
                                                                        output,
                                                                        descriptor,
                                                                        weights,
                                                                        biases,
                                                                        reason);
            break;
        }
        case LayerType::Dequantize:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsDequantizeSupported(input,
                                                              OverrideDataType(output, dataType),
                                                              reason);
            break;
        }
        case LayerType::DetectionPostProcess:
        {
            auto cLayer = PolymorphicDowncast<const DetectionPostProcessLayer*>(&layer);
            const TensorInfo& boxEncodings = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& scores = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& anchors = cLayer->m_Anchors->GetTensorInfo();

            const TensorInfo& detectionBoxes = layer.GetOutputSlot(0).GetTensorInfo();
            const TensorInfo& detectionClasses = layer.GetOutputSlot(1).GetTensorInfo();
            const TensorInfo& detectionScores = layer.GetOutputSlot(2).GetTensorInfo();
            const TensorInfo& numDetections = layer.GetOutputSlot(3).GetTensorInfo();

            const DetectionPostProcessDescriptor& descriptor = cLayer->GetParameters();
            result = layerSupportObject.IsDetectionPostProcessSupported(boxEncodings,
                                                                        scores,
                                                                        anchors,
                                                                        detectionBoxes,
                                                                        detectionClasses,
                                                                        detectionScores,
                                                                        numDetections,
                                                                        descriptor,
                                                                        reason);
            break;
        }
        case LayerType::ElementwiseBinary:
        {
            auto cLayer = PolymorphicDowncast<const ElementwiseBinaryLayer*>(&layer);

            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            std::vector<TensorInfo> infos = { OverrideDataType(input0, dataType),
                                              OverrideDataType(input1, dataType),
                                              OverrideDataType(output, dataType) };
            result = layerSupport->IsLayerSupported(LayerType::ElementwiseBinary,
                                                    infos,
                                                    cLayer->GetParameters(),
                                                    EmptyOptional(),
                                                    EmptyOptional(),
                                                    reason);
            break;
        }
        case LayerType::ElementwiseUnary:
        {
            auto cLayer = PolymorphicDowncast<const ElementwiseUnaryLayer*>(&layer);

            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsElementwiseUnarySupported(OverrideDataType(input, dataType),
                                                                    OverrideDataType(output, dataType),
                                                                    cLayer->GetParameters(),
                                                                    reason);
            break;
        }
        case LayerType::Fill:
        {
            auto cLayer = PolymorphicDowncast<const FillLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            const FillDescriptor& descriptor = cLayer->GetParameters();

            result = layerSupportObject.IsFillSupported(
                OverrideDataType(input, dataType),
                OverrideDataType(output, dataType),
                descriptor,
                reason);
            break;
        }
        case LayerType::FakeQuantization:
        {
            auto cLayer = PolymorphicDowncast<const FakeQuantizationLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = layerSupportObject.IsFakeQuantizationSupported(OverrideDataType(input, dataType),
                                                                    cLayer->GetParameters(),
                                                                    reason);
            break;
        }
        case LayerType::Floor:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsFloorSupported(OverrideDataType(input, dataType),
                                                         OverrideDataType(output, dataType),
                                                         reason);
            break;
        }
        case LayerType::FullyConnected:
        {
            auto cLayer = PolymorphicDowncast<const FullyConnectedLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            const FullyConnectedDescriptor& descriptor = cLayer->GetParameters();
            TensorInfo weightsInfo;
            const TensorInfo* weightsInfoPtr = nullptr;

            weightsInfo = OverrideDataType(layer.GetInputSlot(1).GetConnection()->GetTensorInfo(), dataType);
            weightsInfoPtr = &weightsInfo;

            TensorInfo biasInfo;
            const TensorInfo* biasInfoPtr = nullptr;
            static const TensorInfo dummyBFloat16Bias(TensorShape({1,1,1,1}), DataType::BFloat16);
            static const TensorInfo dummyFloat16Bias(TensorShape({1,1,1,1}), DataType::Float16);
            static const TensorInfo dummyFloat32Bias(TensorShape({1,1,1,1}), DataType::Float32);
            static const TensorInfo dummyQA8Bias(TensorShape({1,1,1,1}), DataType::Signed32);

            if (descriptor.m_BiasEnabled)
            {
                biasInfo = OverrideDataType(layer.GetInputSlot(2).GetConnection()->GetTensorInfo(), dataType);
                biasInfoPtr = &biasInfo;
            }
            else
            {
                // If biases are not enabled pass a dummy tensorinfo for the validation
                switch(input.GetDataType())
                {
                    case DataType::BFloat16:
                    {
                        biasInfoPtr = &dummyBFloat16Bias;
                        break;
                    }
                    case DataType::Float16:
                    {
                        biasInfoPtr = &dummyFloat16Bias;
                        break;
                    }
                    case DataType::Float32:
                    {
                        biasInfoPtr = &dummyFloat32Bias;
                        break;
                    }
                    case DataType::QAsymmU8:
                    case DataType::QAsymmS8:
                    case DataType::QSymmS8:
                    case DataType::QSymmS16:
                    {
                        biasInfoPtr = &dummyQA8Bias;
                        break;
                    }
                    default:
                    {
                        ARMNN_ASSERT_MSG(false, "Unexpected bias type");
                    }
                }
            }
            result = layerSupportObject.IsFullyConnectedSupported(
                                               OverrideDataType(input, dataType),
                                               OverrideDataType(output, dataType),
                                               *weightsInfoPtr,
                                               *biasInfoPtr,
                                               descriptor,
                                               reason);
            break;
        }
        case LayerType::Gather:
        {
            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            auto cLayer = PolymorphicDowncast<const GatherLayer*>(&layer);
            const GatherDescriptor& descriptor = cLayer->GetParameters();
            result = layerSupportObject.IsGatherSupported(OverrideDataType(input0, dataType),
                                                          input1,
                                                          OverrideDataType(output, dataType),
                                                          descriptor,
                                                          reason);
            break;
        }
        case LayerType::GatherNd:
        {
            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsGatherNdSupported(OverrideDataType(input0, dataType),
                                                            input1,
                                                            OverrideDataType(output, dataType),
                                                            reason);
            break;
        }
        case LayerType::Input:
        {
            const TensorInfo& input = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsInputSupported(OverrideDataType(input, dataType), reason);
            break;
        }
        case LayerType::InstanceNormalization:
        {
            auto cLayer = PolymorphicDowncast<const InstanceNormalizationLayer*>(&layer);
            const InstanceNormalizationDescriptor& descriptor = cLayer->GetParameters();

            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsInstanceNormalizationSupported(
                OverrideDataType(input, dataType),
                OverrideDataType(output, dataType),
                descriptor,
                reason);
            break;
        }
        case LayerType::L2Normalization:
        {
            auto cLayer = PolymorphicDowncast<const L2NormalizationLayer*>(&layer);
            const L2NormalizationDescriptor& descriptor = cLayer->GetParameters();

            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsL2NormalizationSupported(
                                                OverrideDataType(input, dataType),
                                                OverrideDataType(output, dataType),
                                                descriptor,
                                                reason);
            break;
        }
        case LayerType::LogicalBinary:
        {
            auto cLayer = PolymorphicDowncast<const LogicalBinaryLayer*>(&layer);

            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsLogicalBinarySupported(input0,
                                                                 input1,
                                                                 output,
                                                                 cLayer->GetParameters(),
                                                                 reason);
            break;
        }
        case LayerType::LogSoftmax:
        {
            auto cLayer = PolymorphicDowncast<const LogSoftmaxLayer*>(&layer);

            const TensorInfo& input  = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsLogSoftmaxSupported(OverrideDataType(input, dataType),
                                                              OverrideDataType(output, dataType),
                                                              cLayer->GetParameters(),
                                                              reason);
            break;
        }
        case LayerType::Lstm:
        {
            auto cLayer = PolymorphicDowncast<const LstmLayer*>(&layer);
            const LstmDescriptor& descriptor = cLayer->GetParameters();

            // All inputs.
            const TensorInfo& input = OverrideDataType(layer.GetInputSlot(0).GetConnection()->GetTensorInfo(),
                                                       dataType);
            const TensorInfo& outputStateIn = OverrideDataType(layer.GetInputSlot(1).GetConnection()->GetTensorInfo(),
                                                               dataType);
            const TensorInfo& cellStateIn = OverrideDataType(layer.GetInputSlot(2).GetConnection()->GetTensorInfo(),
                                                             dataType);
            // All outputs
            const TensorInfo& scratchBuffer = OverrideDataType(layer.GetOutputSlot(0).GetTensorInfo(), dataType);
            const TensorInfo& outputStateOut = OverrideDataType(layer.GetOutputSlot(1).GetTensorInfo(), dataType);
            const TensorInfo& cellStateOut = OverrideDataType(layer.GetOutputSlot(2).GetTensorInfo(), dataType);
            const TensorInfo& output = OverrideDataType(layer.GetOutputSlot(3).GetTensorInfo(), dataType);

            // Basic parameters
            const TensorInfo& inputToForgetWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_InputToForgetWeights->GetTensorInfo(), dataType);
            const TensorInfo& inputToCellWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_InputToCellWeights->GetTensorInfo(), dataType);
            const TensorInfo& inputToOutputWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_InputToOutputWeights->GetTensorInfo(), dataType);
            const TensorInfo& recurrentToForgetWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_RecurrentToForgetWeights->GetTensorInfo(), dataType);
            const TensorInfo& recurrentToCellWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_RecurrentToCellWeights->GetTensorInfo(), dataType);
            const TensorInfo& recurrentToOutputWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_RecurrentToOutputWeights->GetTensorInfo(), dataType);
            const TensorInfo& forgetGateBias
                    = OverrideDataType(cLayer->m_BasicParameters.m_ForgetGateBias->GetTensorInfo(), dataType);
            const TensorInfo& cellBias
                    = OverrideDataType(cLayer->m_BasicParameters.m_CellBias->GetTensorInfo(), dataType);
            const TensorInfo& outputGateBias
                    = OverrideDataType(cLayer->m_BasicParameters.m_OutputGateBias->GetTensorInfo(), dataType);

            LstmInputParamsInfo paramsInfo;

            paramsInfo.m_InputToForgetWeights     = &inputToForgetWeights;
            paramsInfo.m_InputToCellWeights       = &inputToCellWeights;
            paramsInfo.m_InputToOutputWeights     = &inputToOutputWeights;
            paramsInfo.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
            paramsInfo.m_RecurrentToCellWeights   = &recurrentToCellWeights;
            paramsInfo.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
            paramsInfo.m_ForgetGateBias           = &forgetGateBias;
            paramsInfo.m_CellBias                 = &cellBias;
            paramsInfo.m_OutputGateBias           = &outputGateBias;


            // Optional parameters
            TensorInfo optInputToInputWeights;
            TensorInfo optRecurrentToInputWeights;
            TensorInfo optCellToInputWeights;
            TensorInfo optInputGateBias;
            TensorInfo optProjectionWeights;
            TensorInfo optProjectionBias;
            TensorInfo optCellToForgetWeights;
            TensorInfo optCellToOutputWeights;
            TensorInfo optInputLayerNormWeights;
            TensorInfo optForgetLayerNormWeights;
            TensorInfo optCellLayerNormWeights;
            TensorInfo optOutputLayerNormWeights;

            if(!descriptor.m_CifgEnabled)
            {
                optInputToInputWeights =
                    OverrideDataType(cLayer->m_CifgParameters.m_InputToInputWeights->GetTensorInfo(), dataType);
                paramsInfo.m_InputToInputWeights = &optInputToInputWeights;

                optRecurrentToInputWeights =
                    OverrideDataType(cLayer->m_CifgParameters.m_RecurrentToInputWeights->GetTensorInfo(), dataType);
                paramsInfo.m_RecurrentToInputWeights = &optRecurrentToInputWeights;
                optInputGateBias =
                       OverrideDataType(cLayer->m_CifgParameters.m_InputGateBias->GetTensorInfo(), dataType);
                paramsInfo.m_InputGateBias = &optInputGateBias;
            }

            if(descriptor.m_ProjectionEnabled)
            {
                optProjectionWeights =
                    OverrideDataType(cLayer->m_ProjectionParameters.m_ProjectionWeights->GetTensorInfo(), dataType);
                paramsInfo.m_ProjectionWeights = &optProjectionWeights;
                if (cLayer->m_ProjectionParameters.m_ProjectionBias != nullptr)
                {
                    optProjectionBias =
                        OverrideDataType(cLayer->m_ProjectionParameters.m_ProjectionBias->GetTensorInfo(), dataType);
                    paramsInfo.m_ProjectionBias = &optProjectionBias;
                }
            }

            if(descriptor.m_PeepholeEnabled)
            {
                if(!descriptor.m_CifgEnabled)
                {
                    optCellToInputWeights =
                            OverrideDataType(cLayer->m_PeepholeParameters.m_CellToInputWeights->GetTensorInfo(),
                                             dataType);
                    paramsInfo.m_CellToInputWeights = &optCellToInputWeights;
                }
                optCellToForgetWeights =
                    OverrideDataType(cLayer->m_PeepholeParameters.m_CellToForgetWeights->GetTensorInfo(), dataType);
                paramsInfo.m_CellToForgetWeights = &optCellToForgetWeights;
                optCellToOutputWeights =
                    OverrideDataType(cLayer->m_PeepholeParameters.m_CellToOutputWeights->GetTensorInfo(), dataType);
                paramsInfo.m_CellToOutputWeights = &optCellToOutputWeights;
            }

            if(descriptor.m_LayerNormEnabled)
            {
                if (!descriptor.m_CifgEnabled)
                {
                    optInputLayerNormWeights = OverrideDataType(
                            cLayer->m_LayerNormParameters.m_InputLayerNormWeights->GetTensorInfo(), dataType);
                    paramsInfo.m_InputLayerNormWeights = &optInputLayerNormWeights;
                }

                optForgetLayerNormWeights = OverrideDataType(
                        cLayer->m_LayerNormParameters.m_ForgetLayerNormWeights->GetTensorInfo(), dataType);
                paramsInfo.m_ForgetLayerNormWeights = &optForgetLayerNormWeights;

                optCellLayerNormWeights = OverrideDataType(
                        cLayer->m_LayerNormParameters.m_CellLayerNormWeights->GetTensorInfo(), dataType);
                paramsInfo.m_CellLayerNormWeights = &optCellLayerNormWeights;

                optOutputLayerNormWeights = OverrideDataType(
                        cLayer->m_LayerNormParameters.m_OutputLayerNormWeights->GetTensorInfo(), dataType);
                paramsInfo.m_OutputLayerNormWeights = &optOutputLayerNormWeights;
            }

            result = layerSupportObject.IsLstmSupported(
                                     input,
                                     outputStateIn,
                                     cellStateIn,
                                     scratchBuffer,
                                     outputStateOut,
                                     cellStateOut,
                                     output,
                                     descriptor,
                                     paramsInfo,
                                     reason);
            break;
        }
        case LayerType::Maximum:
        {
            ARMNN_NO_DEPRECATE_WARN_BEGIN
            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsMaximumSupported(OverrideDataType(input0, dataType),
                                                           OverrideDataType(input1, dataType),
                                                           OverrideDataType(output, dataType),
                                                           reason);
            ARMNN_NO_DEPRECATE_WARN_END
            break;
        }
        case LayerType::MemCopy:
        {
            const TensorInfo& input  = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsMemCopySupported(OverrideDataType(input, dataType),
                                                           OverrideDataType(output, dataType),
                                                           reason);
            break;
        }
        case LayerType::MemImport:
        {
            const TensorInfo& input  = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsMemImportSupported(OverrideDataType(input, dataType),
                                                             OverrideDataType(output, dataType),
                                                             reason);
            break;
        }
        case LayerType::Merge:
        {
            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsMergeSupported(OverrideDataType(input0, dataType),
                                                         OverrideDataType(input1, dataType),
                                                         OverrideDataType(output, dataType),
                                                         reason);
            break;
        }
        case LayerType::Concat:
        {
            auto cLayer = PolymorphicDowncast<const ConcatLayer*>(&layer);

            // Get vector of all inputs.
            auto getTensorInfo = [&dataType](const InputSlot& slot)
                {
                    return OverrideDataType(slot.GetConnectedOutputSlot()->GetTensorInfo(), dataType);
                };

            auto beginI = MakeTransformIterator(layer.GetInputSlots().begin(), getTensorInfo);
            auto endI = MakeTransformIterator(layer.GetInputSlots().end(), getTensorInfo);
            std::vector<TensorInfo> inputs(beginI, endI);

            auto getTensorInfoPtr = [](const TensorInfo& info)
                {
                    return &info;
                };

            auto beginPtr = MakeTransformIterator(inputs.begin(), getTensorInfoPtr);
            auto endPtr = MakeTransformIterator(inputs.end(), getTensorInfoPtr);
            std::vector<const TensorInfo*> inputPtrs(beginPtr, endPtr);

            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsConcatSupported(inputPtrs, output, cLayer->GetParameters(), reason);


            break;
        }
        case LayerType::Multiplication:
        {
            ARMNN_NO_DEPRECATE_WARN_BEGIN
            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsMultiplicationSupported(
                                               OverrideDataType(input0, dataType),
                                               OverrideDataType(input1, dataType),
                                               OverrideDataType(output, dataType),
                                               reason);
            ARMNN_NO_DEPRECATE_WARN_END
            break;
        }
        case LayerType::Normalization:
        {
            auto cLayer = PolymorphicDowncast<const NormalizationLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsNormalizationSupported(OverrideDataType(input, dataType),
                                                                 OverrideDataType(output, dataType),
                                                                 cLayer->GetParameters(),
                                                                 reason);
            break;
        }
        case LayerType::Output:
        {
            const TensorInfo& output = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = layerSupportObject.IsOutputSupported(OverrideDataType(output, dataType), reason);
            break;
        }
        case LayerType::Permute:
        {
            auto cLayer = PolymorphicDowncast<const PermuteLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsPermuteSupported(OverrideDataType(input, dataType),
                                                           OverrideDataType(output, dataType),
                                                           cLayer->GetParameters(),
                                                           reason);
            break;
        }
        case LayerType::Pad:
        {
            auto cLayer = PolymorphicDowncast<const PadLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsPadSupported(
                                    OverrideDataType(input, dataType),
                                    OverrideDataType(output, dataType),
                                    cLayer->GetParameters(),
                                    reason);
            break;
        }
        case LayerType::Pooling2d:
        {
            auto cLayer = PolymorphicDowncast<const Pooling2dLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsPooling2dSupported(OverrideDataType(input, dataType),
                                                             OverrideDataType(output, dataType),
                                                             cLayer->GetParameters(),
                                                             reason);
            break;
        }
        case LayerType::Pooling3d:
        {
            auto cLayer = PolymorphicDowncast<const Pooling3dLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsPooling3dSupported(OverrideDataType(input, dataType),
                                                             OverrideDataType(output, dataType),
                                                             cLayer->GetParameters(),
                                                             reason);
            break;
        }
        case LayerType::PreCompiled:
        {
            auto cLayer = PolymorphicDowncast<const PreCompiledLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = layerSupportObject.IsPreCompiledSupported(OverrideDataType(input, dataType),
                                                               cLayer->GetParameters(),
                                                               reason);
            break;
        }
        case LayerType::Quantize:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsQuantizeSupported(input, output, reason);
            break;
        }
        case LayerType::QLstm:
        {
            auto cLayer = PolymorphicDowncast<const QLstmLayer*>(&layer);
            const QLstmDescriptor& descriptor = cLayer->GetParameters();

            // Inputs
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& previousOutputIn = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& previousCellStateIn = layer.GetInputSlot(2).GetConnection()->GetTensorInfo();

            // Outputs
            const TensorInfo& outputStateOut = layer.GetOutputSlot(0).GetTensorInfo();
            const TensorInfo& cellStateOut = layer.GetOutputSlot(1).GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(2).GetTensorInfo();

            // Lstm parameters
            LstmInputParamsInfo paramsInfo;

            // Basic parameters
            ARMNN_ASSERT(cLayer->m_BasicParameters.m_InputToForgetWeights.get() != nullptr);
            ARMNN_ASSERT(cLayer->m_BasicParameters.m_InputToCellWeights.get() != nullptr);
            ARMNN_ASSERT(cLayer->m_BasicParameters.m_InputToOutputWeights.get() != nullptr);
            paramsInfo.m_InputToForgetWeights = &cLayer->m_BasicParameters.m_InputToForgetWeights->GetTensorInfo();
            paramsInfo.m_InputToCellWeights   = &cLayer->m_BasicParameters.m_InputToCellWeights->GetTensorInfo();
            paramsInfo.m_InputToOutputWeights = &cLayer->m_BasicParameters.m_InputToOutputWeights->GetTensorInfo();

            paramsInfo.m_RecurrentToForgetWeights =
                    &cLayer->m_BasicParameters.m_RecurrentToForgetWeights->GetTensorInfo();
            paramsInfo.m_RecurrentToCellWeights   =
                    &cLayer->m_BasicParameters.m_RecurrentToCellWeights->GetTensorInfo();
            paramsInfo.m_RecurrentToOutputWeights =
                    &cLayer->m_BasicParameters.m_RecurrentToOutputWeights->GetTensorInfo();

            paramsInfo.m_ForgetGateBias = &cLayer->m_BasicParameters.m_ForgetGateBias->GetTensorInfo();
            paramsInfo.m_CellBias       = &cLayer->m_BasicParameters.m_CellBias->GetTensorInfo();
            paramsInfo.m_OutputGateBias = &cLayer->m_BasicParameters.m_OutputGateBias->GetTensorInfo();

            if(!descriptor.m_CifgEnabled)
            {
                paramsInfo.m_InputToInputWeights = &cLayer->m_CifgParameters.m_InputToInputWeights->GetTensorInfo();
                paramsInfo.m_RecurrentToInputWeights =
                        &cLayer->m_CifgParameters.m_RecurrentToInputWeights->GetTensorInfo();
                paramsInfo.m_InputGateBias = &cLayer->m_CifgParameters.m_InputGateBias->GetTensorInfo();
            }

            if(descriptor.m_ProjectionEnabled)
            {
                paramsInfo.m_ProjectionWeights = &cLayer->m_ProjectionParameters.m_ProjectionWeights->GetTensorInfo();

                // Projection bias is optional even if projection is enabled
                if (cLayer->m_ProjectionParameters.m_ProjectionBias != nullptr)
                {
                    paramsInfo.m_ProjectionBias = &cLayer->m_ProjectionParameters.m_ProjectionBias->GetTensorInfo();
                }
            }

            if(descriptor.m_PeepholeEnabled)
            {
                if (!descriptor.m_CifgEnabled)
                {
                    paramsInfo.m_CellToInputWeights =
                            &cLayer->m_PeepholeParameters.m_CellToInputWeights->GetTensorInfo();
                }

                paramsInfo.m_CellToForgetWeights =
                        &cLayer->m_PeepholeParameters.m_CellToForgetWeights->GetTensorInfo();
                paramsInfo.m_CellToOutputWeights = &cLayer->m_PeepholeParameters.m_CellToOutputWeights->GetTensorInfo();
            }

            if(descriptor.m_LayerNormEnabled)
            {
                if (!descriptor.m_CifgEnabled)
                {
                    paramsInfo.m_InputLayerNormWeights =
                            &cLayer->m_LayerNormParameters.m_InputLayerNormWeights->GetTensorInfo();
                }

                paramsInfo.m_ForgetLayerNormWeights =
                        &cLayer->m_LayerNormParameters.m_ForgetLayerNormWeights->GetTensorInfo();
                paramsInfo.m_CellLayerNormWeights =
                        &cLayer->m_LayerNormParameters.m_CellLayerNormWeights->GetTensorInfo();
                paramsInfo.m_OutputLayerNormWeights =
                        &cLayer->m_LayerNormParameters.m_OutputLayerNormWeights->GetTensorInfo();
            }

            result = layerSupportObject.IsQLstmSupported(input,
                                                         previousOutputIn,
                                                         previousCellStateIn,
                                                         outputStateOut,
                                                         cellStateOut,
                                                         output,
                                                         descriptor,
                                                         paramsInfo,
                                                         reason);
            break;
        }
        case LayerType::QuantizedLstm:
        {
            auto cLayer = PolymorphicDowncast<const QuantizedLstmLayer*>(&layer);

            // Inputs
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& previousCellStateIn = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& previousOutputIn = layer.GetInputSlot(2).GetConnection()->GetTensorInfo();

            // Outputs
            const TensorInfo& cellStateOut = layer.GetOutputSlot(0).GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(1).GetTensorInfo();

            // QuantizedLstm parameters
            QuantizedLstmInputParamsInfo paramsInfo;

            paramsInfo.m_InputToInputWeights      =
                    &cLayer->m_QuantizedLstmParameters.m_InputToInputWeights->GetTensorInfo();
            paramsInfo.m_InputToForgetWeights     =
                    &cLayer->m_QuantizedLstmParameters.m_InputToForgetWeights->GetTensorInfo();
            paramsInfo.m_InputToCellWeights       =
                    &cLayer->m_QuantizedLstmParameters.m_InputToCellWeights->GetTensorInfo();
            paramsInfo.m_InputToOutputWeights     =
                    &cLayer->m_QuantizedLstmParameters.m_InputToOutputWeights->GetTensorInfo();

            paramsInfo.m_RecurrentToInputWeights  =
                    &cLayer->m_QuantizedLstmParameters.m_RecurrentToInputWeights->GetTensorInfo();
            paramsInfo.m_RecurrentToForgetWeights =
                    &cLayer->m_QuantizedLstmParameters.m_RecurrentToForgetWeights->GetTensorInfo();
            paramsInfo.m_RecurrentToCellWeights   =
                    &cLayer->m_QuantizedLstmParameters.m_RecurrentToCellWeights->GetTensorInfo();
            paramsInfo.m_RecurrentToOutputWeights =
                    &cLayer->m_QuantizedLstmParameters.m_RecurrentToOutputWeights->GetTensorInfo();

            paramsInfo.m_InputGateBias            =
                    &cLayer->m_QuantizedLstmParameters.m_InputGateBias->GetTensorInfo();
            paramsInfo.m_ForgetGateBias           =
                    &cLayer->m_QuantizedLstmParameters.m_ForgetGateBias->GetTensorInfo();
            paramsInfo.m_CellBias                 =
                    &cLayer->m_QuantizedLstmParameters.m_CellBias->GetTensorInfo();
            paramsInfo.m_OutputGateBias           =
                    &cLayer->m_QuantizedLstmParameters.m_OutputGateBias->GetTensorInfo();;

            result = layerSupportObject.IsQuantizedLstmSupported(input,
                                                                 previousCellStateIn,
                                                                 previousOutputIn,
                                                                 cellStateOut,
                                                                 output,
                                                                 paramsInfo,
                                                                 reason);
            break;
        }
        case LayerType::Division:
        {
            ARMNN_NO_DEPRECATE_WARN_BEGIN
            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsDivisionSupported(
                                         OverrideDataType(input0, dataType),
                                         OverrideDataType(input1, dataType),
                                         OverrideDataType(output, dataType),
                                         reason);
            ARMNN_NO_DEPRECATE_WARN_END
            break;
        }
        case LayerType::Rank:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsRankSupported(OverrideDataType(input, dataType),
                                                        OverrideDataType(output, dataType),
                                                        reason);
            break;
        }
        case LayerType::Reshape:
        {
            auto cLayer = PolymorphicDowncast<const ReshapeLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsReshapeSupported(OverrideDataType(input, dataType),
                                                           OverrideDataType(output, dataType),
                                                           cLayer->GetParameters(),
                                                           reason);
            break;
        }
        case LayerType::Resize:
        {
            auto cLayer = PolymorphicDowncast<const ResizeLayer*>(&layer);
            const TensorInfo& input  = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsResizeSupported(OverrideDataType(input, dataType),
                                                          OverrideDataType(output, dataType),
                                                          cLayer->GetParameters(),
                                                          reason);
            break;
        }
        case LayerType::Shape:
        {
            const TensorInfo& input  = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsShapeSupported(OverrideDataType(input, dataType),
                                                         OverrideDataType(output, dataType),
                                                         reason);
            break;
        }
        case LayerType::Slice:
        {
            auto cLayer = PolymorphicDowncast<const SliceLayer*>(&layer);

            const TensorInfo& input  = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsSliceSupported(OverrideDataType(input, dataType),
                                                         OverrideDataType(output, dataType),
                                                         cLayer->GetParameters(),
                                                         reason);
            break;
        }
        case LayerType::Softmax:
        {
            auto cLayer = PolymorphicDowncast<const SoftmaxLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsSoftmaxSupported(OverrideDataType(input, dataType),
                                                           OverrideDataType(output, dataType),
                                                           cLayer->GetParameters(),
                                                           reason);
            break;
        }
        case LayerType::SpaceToBatchNd:
        {
            auto cLayer = PolymorphicDowncast<const SpaceToBatchNdLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsSpaceToBatchNdSupported(OverrideDataType(input, dataType),
                                                                  OverrideDataType(output, dataType),
                                                                  cLayer->GetParameters(),
                                                                  reason);
            break;
        }
        case LayerType::SpaceToDepth:
        {
            auto cLayer = PolymorphicDowncast<const SpaceToDepthLayer*>(&layer);

            const TensorInfo& input  = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsSpaceToDepthSupported(OverrideDataType(input, dataType),
                                                                OverrideDataType(output, dataType),
                                                                cLayer->GetParameters(),
                                                                reason);
            break;
        }
        case LayerType::Splitter:
        {
            auto cLayer = PolymorphicDowncast<const SplitterLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();

            // Get vector of all outputs.
            auto getTensorInfo = [&dataType](const OutputSlot& slot)
            {
                return OverrideDataType(slot.GetTensorInfo(), dataType);
            };
            auto beginI = MakeTransformIterator(layer.GetOutputSlots().begin(), getTensorInfo);
            auto endI = MakeTransformIterator(layer.GetOutputSlots().end(), getTensorInfo);
            std::vector<TensorInfo> outputs(beginI, endI);

            const std::vector<std::reference_wrapper<TensorInfo>> outputPtrs(outputs.begin(), outputs.end());

            result = layerSupportObject.IsSplitterSupported(OverrideDataType(input, dataType),
                                                            outputPtrs,
                                                            cLayer->GetParameters(),
                                                            reason);
            break;
        }
        case LayerType::Stack:
        {
            auto cLayer = PolymorphicDowncast<const StackLayer*>(&layer);

            // Get vector of all inputs.
            auto getTensorInfo = [&dataType](const InputSlot& slot)
                {
                    return OverrideDataType(slot.GetConnectedOutputSlot()->GetTensorInfo(), dataType);
                };
            auto beginI = MakeTransformIterator(layer.GetInputSlots().begin(), getTensorInfo);
            auto endI = MakeTransformIterator(layer.GetInputSlots().end(), getTensorInfo);
            std::vector<TensorInfo> inputs(beginI, endI);

            auto getTensorInfoPtr = [](const TensorInfo& info)
                {
                    return &info;
                };
            auto beginPtr = MakeTransformIterator(inputs.begin(), getTensorInfoPtr);
            auto endPtr = MakeTransformIterator(inputs.end(), getTensorInfoPtr);
            std::vector<const TensorInfo*> inputPtrs(beginPtr, endPtr);

            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsStackSupported(inputPtrs, output, cLayer->GetParameters(), reason);

            break;
        }
        case LayerType::StandIn:
        {
            auto cLayer = PolymorphicDowncast<const StandInLayer*>(&layer);

            // Get vector of all inputs.
            auto getTensorInfoIn = [&dataType](const InputSlot& slot)
                {
                    return OverrideDataType(slot.GetConnectedOutputSlot()->GetTensorInfo(), dataType);
                };
            auto getTensorInfoOut = [&dataType](const OutputSlot& slot)
                {
                    return OverrideDataType(slot.GetTensorInfo(), dataType);
                };
            auto beginI = MakeTransformIterator(layer.GetInputSlots().begin(), getTensorInfoIn);
            auto endI = MakeTransformIterator(layer.GetInputSlots().end(), getTensorInfoIn);
            std::vector<TensorInfo> inputs(beginI, endI);

            auto beginO = MakeTransformIterator(layer.GetOutputSlots().begin(), getTensorInfoOut);
            auto endO = MakeTransformIterator(layer.GetOutputSlots().end(), getTensorInfoOut);
            std::vector<TensorInfo> outputs(beginO, endO);


            auto getTensorInfoPtr = [](const TensorInfo& info)
                {
                    return &info;
                };
            auto beginPtrI = MakeTransformIterator(inputs.begin(), getTensorInfoPtr);
            auto endPtrI = MakeTransformIterator(inputs.end(), getTensorInfoPtr);
            std::vector<const TensorInfo*> inputPtrs(beginPtrI, endPtrI);

            auto beginPtrO = MakeTransformIterator(outputs.begin(), getTensorInfoPtr);
            auto endPtrO = MakeTransformIterator(outputs.end(), getTensorInfoPtr);
            std::vector<const TensorInfo*> outputPtrs(beginPtrO, endPtrO);


            result = layerSupportObject.IsStandInSupported(inputPtrs,
                                                           outputPtrs,
                                                           cLayer->GetParameters(),
                                                           reason);
            break;
        }
        case LayerType::StridedSlice:
        {
            auto cLayer = PolymorphicDowncast<const StridedSliceLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsStridedSliceSupported(OverrideDataType(input, dataType),
                                                                OverrideDataType(output, dataType),
                                                                cLayer->GetParameters(),
                                                                reason);
            break;
        }
        case LayerType::Subtraction:
        {
            ARMNN_NO_DEPRECATE_WARN_BEGIN
            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsSubtractionSupported(
                                            OverrideDataType(input0, dataType),
                                            OverrideDataType(input1, dataType),
                                            OverrideDataType(output, dataType),
                                            reason);
            ARMNN_NO_DEPRECATE_WARN_END
            break;
        }
        case LayerType::Switch:
        {
            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output0 = layer.GetOutputSlot(0).GetTensorInfo();
            const TensorInfo& output1 = layer.GetOutputSlot(1).GetTensorInfo();
            result = layerSupportObject.IsSwitchSupported(OverrideDataType(input0, dataType),
                                                          OverrideDataType(input1, dataType),
                                                          OverrideDataType(output0, dataType),
                                                          OverrideDataType(output1, dataType),
                                                          reason);
            break;
        }
        case LayerType::Mean:
        {
            auto cLayer = PolymorphicDowncast<const MeanLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsMeanSupported(
                                     OverrideDataType(input, dataType),
                                     OverrideDataType(output, dataType),
                                     cLayer->GetParameters(),
                                     reason);
            break;
        }
        case LayerType::Minimum:
        {
            ARMNN_NO_DEPRECATE_WARN_BEGIN
            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsMinimumSupported(OverrideDataType(input0, dataType),
                                                           OverrideDataType(input1, dataType),
                                                           OverrideDataType(output, dataType),
                                                           reason);
            ARMNN_NO_DEPRECATE_WARN_END
            break;
        }
        case LayerType::Prelu:
        {
            const TensorInfo& input  = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& alpha  = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsPreluSupported(OverrideDataType(input,  dataType),
                                                         OverrideDataType(alpha,  dataType),
                                                         OverrideDataType(output, dataType),
                                                         reason);
            break;
        }
        case LayerType::Transpose:
        {
            auto cLayer = PolymorphicDowncast<const TransposeLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = layerSupportObject.IsTransposeSupported(OverrideDataType(input, dataType),
                                                             OverrideDataType(output, dataType),
                                                             cLayer->GetParameters(),
                                                             reason);
            break;
        }
        case LayerType::TransposeConvolution2d:
        {
            auto cLayer = PolymorphicDowncast<const TransposeConvolution2dLayer*>(&layer);

            const TensorInfo input  = OverrideDataType(layer.GetInputSlot(0).GetConnection()->GetTensorInfo(),
                                                       dataType);
            const TensorInfo output = OverrideDataType(layer.GetOutputSlot(0).GetTensorInfo(), dataType);

            const TransposeConvolution2dDescriptor& descriptor  = cLayer->GetParameters();

            Optional<TensorInfo> biases;
            if (descriptor.m_BiasEnabled)
            {
                ARMNN_ASSERT(cLayer->m_Bias.get() != nullptr);
                biases = OverrideDataType(cLayer->m_Bias->GetTensorInfo(),
                                          GetBiasTypeFromWeightsType(dataType));
            }

            ARMNN_ASSERT(cLayer->m_Weight.get() != nullptr);
            const TensorInfo weights = OverrideDataType(cLayer->m_Weight->GetTensorInfo(), dataType);

            result = layerSupportObject.IsTransposeConvolution2dSupported(input,
                                                                          output,
                                                                          descriptor,
                                                                          weights,
                                                                          biases,
                                                                          reason);

            break;
        }
        case LayerType::Reduce:
        {
            auto cLayer = PolymorphicDowncast<const ReduceLayer*>(&layer);
            const TensorInfo& input  = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();

            result = layerSupportObject.IsReduceSupported(OverrideDataType(input, dataType),
                                                          OverrideDataType(output, dataType),
                                                          cLayer->GetParameters(),
                                                          reason);
            break;
        }
        case LayerType::UnidirectionalSequenceLstm:
        {
            auto cLayer = PolymorphicDowncast<const UnidirectionalSequenceLstmLayer*>(&layer);
            const UnidirectionalSequenceLstmDescriptor& descriptor = cLayer->GetParameters();

            // All inputs.
            const TensorInfo& input = OverrideDataType(layer.GetInputSlot(0).GetConnection()->GetTensorInfo(),
                                                       dataType);
            const TensorInfo& outputStateIn = OverrideDataType(layer.GetInputSlot(1).GetConnection()->GetTensorInfo(),
                                                               dataType);
            const TensorInfo& cellStateIn = OverrideDataType(layer.GetInputSlot(2).GetConnection()->GetTensorInfo(),
                                                             dataType);
            // Outputs
            const TensorInfo& outputStateOut = OverrideDataType(layer.GetOutputSlot(0).GetTensorInfo(), dataType);
            const TensorInfo& cellStateOut = OverrideDataType(layer.GetOutputSlot(1).GetTensorInfo(), dataType);
            const TensorInfo& output = OverrideDataType(layer.GetOutputSlot(2).GetTensorInfo(), dataType);

            // Basic parameters
            const TensorInfo& inputToForgetWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_InputToForgetWeights->GetTensorInfo(), dataType);
            const TensorInfo& inputToCellWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_InputToCellWeights->GetTensorInfo(), dataType);
            const TensorInfo& inputToOutputWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_InputToOutputWeights->GetTensorInfo(), dataType);
            const TensorInfo& recurrentToForgetWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_RecurrentToForgetWeights->GetTensorInfo(), dataType);
            const TensorInfo& recurrentToCellWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_RecurrentToCellWeights->GetTensorInfo(), dataType);
            const TensorInfo& recurrentToOutputWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_RecurrentToOutputWeights->GetTensorInfo(), dataType);
            const TensorInfo& forgetGateBias
                    = OverrideDataType(cLayer->m_BasicParameters.m_ForgetGateBias->GetTensorInfo(), dataType);
            const TensorInfo& cellBias
                    = OverrideDataType(cLayer->m_BasicParameters.m_CellBias->GetTensorInfo(), dataType);
            const TensorInfo& outputGateBias
                    = OverrideDataType(cLayer->m_BasicParameters.m_OutputGateBias->GetTensorInfo(), dataType);

            LstmInputParamsInfo paramsInfo;

            paramsInfo.m_InputToForgetWeights     = &inputToForgetWeights;
            paramsInfo.m_InputToCellWeights       = &inputToCellWeights;
            paramsInfo.m_InputToOutputWeights     = &inputToOutputWeights;
            paramsInfo.m_RecurrentToForgetWeights = &recurrentToForgetWeights;
            paramsInfo.m_RecurrentToCellWeights   = &recurrentToCellWeights;
            paramsInfo.m_RecurrentToOutputWeights = &recurrentToOutputWeights;
            paramsInfo.m_ForgetGateBias           = &forgetGateBias;
            paramsInfo.m_CellBias                 = &cellBias;
            paramsInfo.m_OutputGateBias           = &outputGateBias;

            // Optional parameters
            TensorInfo optInputToInputWeights;
            TensorInfo optRecurrentToInputWeights;
            TensorInfo optCellToInputWeights;
            TensorInfo optInputGateBias;
            TensorInfo optProjectionWeights;
            TensorInfo optProjectionBias;
            TensorInfo optCellToForgetWeights;
            TensorInfo optCellToOutputWeights;
            TensorInfo optInputLayerNormWeights;
            TensorInfo optForgetLayerNormWeights;
            TensorInfo optCellLayerNormWeights;
            TensorInfo optOutputLayerNormWeights;

            if(!descriptor.m_CifgEnabled)
            {
                optInputToInputWeights =
                    OverrideDataType(cLayer->m_CifgParameters.m_InputToInputWeights->GetTensorInfo(), dataType);
                paramsInfo.m_InputToInputWeights = &optInputToInputWeights;

                optRecurrentToInputWeights =
                    OverrideDataType(cLayer->m_CifgParameters.m_RecurrentToInputWeights->GetTensorInfo(), dataType);
                paramsInfo.m_RecurrentToInputWeights = &optRecurrentToInputWeights;
                optInputGateBias =
                       OverrideDataType(cLayer->m_CifgParameters.m_InputGateBias->GetTensorInfo(), dataType);
                paramsInfo.m_InputGateBias = &optInputGateBias;
            }

            if(descriptor.m_ProjectionEnabled)
            {
                optProjectionWeights =
                    OverrideDataType(cLayer->m_ProjectionParameters.m_ProjectionWeights->GetTensorInfo(), dataType);
                paramsInfo.m_ProjectionWeights = &optProjectionWeights;
                if (cLayer->m_ProjectionParameters.m_ProjectionBias != nullptr)
                {
                    optProjectionBias =
                        OverrideDataType(cLayer->m_ProjectionParameters.m_ProjectionBias->GetTensorInfo(), dataType);
                    paramsInfo.m_ProjectionBias = &optProjectionBias;
                }
            }

            if(descriptor.m_PeepholeEnabled)
            {
                if(!descriptor.m_CifgEnabled)
                {
                    optCellToInputWeights =
                            OverrideDataType(cLayer->m_PeepholeParameters.m_CellToInputWeights->GetTensorInfo(),
                                             dataType);
                    paramsInfo.m_CellToInputWeights = &optCellToInputWeights;
                }
                optCellToForgetWeights =
                    OverrideDataType(cLayer->m_PeepholeParameters.m_CellToForgetWeights->GetTensorInfo(), dataType);
                paramsInfo.m_CellToForgetWeights = &optCellToForgetWeights;
                optCellToOutputWeights =
                    OverrideDataType(cLayer->m_PeepholeParameters.m_CellToOutputWeights->GetTensorInfo(), dataType);
                paramsInfo.m_CellToOutputWeights = &optCellToOutputWeights;
            }

            if(descriptor.m_LayerNormEnabled)
            {
                if (!descriptor.m_CifgEnabled)
                {
                    optInputLayerNormWeights = OverrideDataType(
                            cLayer->m_LayerNormParameters.m_InputLayerNormWeights->GetTensorInfo(), dataType);
                    paramsInfo.m_InputLayerNormWeights = &optInputLayerNormWeights;
                }

                optForgetLayerNormWeights = OverrideDataType(
                        cLayer->m_LayerNormParameters.m_ForgetLayerNormWeights->GetTensorInfo(), dataType);
                paramsInfo.m_ForgetLayerNormWeights = &optForgetLayerNormWeights;

                optCellLayerNormWeights = OverrideDataType(
                        cLayer->m_LayerNormParameters.m_CellLayerNormWeights->GetTensorInfo(), dataType);
                paramsInfo.m_CellLayerNormWeights = &optCellLayerNormWeights;

                optOutputLayerNormWeights = OverrideDataType(
                        cLayer->m_LayerNormParameters.m_OutputLayerNormWeights->GetTensorInfo(), dataType);
                paramsInfo.m_OutputLayerNormWeights = &optOutputLayerNormWeights;
            }

            result = layerSupportObject.IsUnidirectionalSequenceLstmSupported(input,
                                                                              outputStateIn,
                                                                              cellStateIn,
                                                                              outputStateOut,
                                                                              cellStateOut,
                                                                              output,
                                                                              descriptor,
                                                                              paramsInfo,
                                                                              reason);
            break;
        }
        default:
        {
            ARMNN_ASSERT_MSG(false, "WorkloadFactory did not recognise type of layer.");
            reason.value() = "Unrecognised layer type";
            result = false;
            break;
        }
    }
    return result;
}

bool IWorkloadFactory::IsLayerSupported(const BackendId& backendId,
                                        const IConnectableLayer& connectableLayer,
                                        Optional<DataType> dataType,
                                        std::string& outReasonIfUnsupported)
{
    return IsLayerConfigurationSupported(backendId, connectableLayer, dataType, outReasonIfUnsupported);
}

bool IWorkloadFactory::IsLayerSupported(const IConnectableLayer& connectableLayer,
                                        Optional<DataType> dataType,
                                        std::string& outReasonIfUnsupported)
{
    auto layer = PolymorphicDowncast<const Layer*>(&connectableLayer);
    return IsLayerConfigurationSupported(layer->GetBackendId(), connectableLayer, dataType, outReasonIfUnsupported);
}

// TODO merge with defaulted modelOptions above
bool IWorkloadFactory::IsLayerSupported(const IConnectableLayer& connectableLayer,
                                        Optional<DataType> dataType,
                                        std::string& outReasonIfUnsupported,
                                        const ModelOptions& modelOptions)
{
    auto layer = PolymorphicDowncast<const Layer*>(&connectableLayer);
    return IsLayerConfigurationSupported(layer->GetBackendId(),
                                         connectableLayer,
                                         dataType,
                                         outReasonIfUnsupported,
                                         modelOptions);
}

bool IWorkloadFactory::IsLayerSupported(const BackendId& backendId,
                                        const IConnectableLayer& connectableLayer,
                                        Optional<DataType> dataType,
                                        std::string& outReasonIfUnsupported,
                                        const ModelOptions& modelOptions)
{
    return IsLayerConfigurationSupported(backendId,
                                         connectableLayer,
                                         dataType,
                                         outReasonIfUnsupported,
                                         modelOptions);
}
ARMNN_NO_DEPRECATE_WARN_BEGIN
std::unique_ptr<IWorkload> IWorkloadFactory::CreateWorkload(LayerType type,
                                                            const QueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    switch(type)
    {
        case LayerType::Activation :
        {
            auto activationQueueDescriptor = PolymorphicDowncast<const ActivationQueueDescriptor*>(&descriptor);
            return CreateActivation(*activationQueueDescriptor, info);
        }
        case LayerType::Addition :
        {
            auto additionQueueDescriptor = PolymorphicDowncast<const AdditionQueueDescriptor*>(&descriptor);
            return CreateAddition(*additionQueueDescriptor, info);
        }
        case LayerType::ArgMinMax :
        {
            auto argMinMaxQueueDescriptor = PolymorphicDowncast<const ArgMinMaxQueueDescriptor*>(&descriptor);
            return CreateArgMinMax(*argMinMaxQueueDescriptor, info);
        }
        case LayerType::BatchNormalization :
        {
            auto batchNormQueueDescriptor = PolymorphicDowncast<const BatchNormalizationQueueDescriptor*>(&descriptor);
            return CreateBatchNormalization(*batchNormQueueDescriptor, info);
        }
        case LayerType::BatchToSpaceNd :
        {
            auto batchToSpaceNdQueueDescriptor
                    = PolymorphicDowncast<const BatchToSpaceNdQueueDescriptor*>(&descriptor);
            return CreateBatchToSpaceNd(*batchToSpaceNdQueueDescriptor, info);
        }
        case LayerType::Cast :
        {
            auto castQueueDescriptor = PolymorphicDowncast<const CastQueueDescriptor*>(&descriptor);
            return CreateCast(*castQueueDescriptor, info);
        }
        case LayerType::ChannelShuffle :
        {
            auto channelShuffleQueueDescriptor
                    = PolymorphicDowncast<const ChannelShuffleQueueDescriptor*>(&descriptor);
            return CreateChannelShuffle(*channelShuffleQueueDescriptor, info);
        }
        case LayerType::Comparison :
        {
            auto comparisonQueueDescriptor = PolymorphicDowncast<const ComparisonQueueDescriptor*>(&descriptor);
            return CreateComparison(*comparisonQueueDescriptor, info);
        }
        case LayerType::Concat :
        {
            auto concatQueueDescriptor = PolymorphicDowncast<const ConcatQueueDescriptor*>(&descriptor);
            return CreateConcat(*concatQueueDescriptor, info);
        }
        case LayerType::Constant :
        {
            auto constantQueueDescriptor = PolymorphicDowncast<const ConstantQueueDescriptor*>(&descriptor);
            return CreateConstant(*constantQueueDescriptor, info);
        }
        case LayerType::ConvertFp16ToFp32:
        {
            auto convertFp16ToFp32QueueDescriptor
                    = PolymorphicDowncast<const ConvertFp16ToFp32QueueDescriptor*>(&descriptor);
            return CreateConvertFp16ToFp32(*convertFp16ToFp32QueueDescriptor, info);
        }
        case LayerType::ConvertFp32ToFp16:
        {
            auto convertFp32ToFp16QueueDescriptor
                    = PolymorphicDowncast<const ConvertFp32ToFp16QueueDescriptor*>(&descriptor);
            return CreateConvertFp32ToFp16(*convertFp32ToFp16QueueDescriptor, info);
        }
        case LayerType::Convolution2d:
        {
            auto convolution2dQueueDescriptor = PolymorphicDowncast<const Convolution2dQueueDescriptor*>(&descriptor);
            return CreateConvolution2d(*convolution2dQueueDescriptor, info);
        }
        case LayerType::Convolution3d:
        {
            auto convolution3dQueueDescriptor = PolymorphicDowncast<const Convolution3dQueueDescriptor*>(&descriptor);
            return CreateConvolution3d(*convolution3dQueueDescriptor, info);
        }
        case LayerType::Debug:
        {
            auto debugQueueDescriptor = PolymorphicDowncast<const DebugQueueDescriptor*>(&descriptor);
            return CreateDebug(*debugQueueDescriptor, info);
        }
        case LayerType::DepthToSpace:
        {
            auto depthToSpaceQueueDescriptor = PolymorphicDowncast<const DepthToSpaceQueueDescriptor*>(&descriptor);
            return CreateDepthToSpace(*depthToSpaceQueueDescriptor, info);
        }
        case LayerType::DepthwiseConvolution2d:
        {
            auto depthwiseConvolution2DQueueDescriptor
                    = PolymorphicDowncast<const DepthwiseConvolution2dQueueDescriptor*>(&descriptor);
            return CreateDepthwiseConvolution2d(*depthwiseConvolution2DQueueDescriptor, info);
        }
        case LayerType::Dequantize:
        {
            auto dequantizeQueueDescriptor = PolymorphicDowncast<const DequantizeQueueDescriptor*>(&descriptor);
            return CreateDequantize(*dequantizeQueueDescriptor, info);
        }
        case LayerType::DetectionPostProcess:
        {
            auto detectionPostProcessQueueDescriptor
                    = PolymorphicDowncast<const DetectionPostProcessQueueDescriptor*>(&descriptor);
            return CreateDetectionPostProcess(*detectionPostProcessQueueDescriptor, info);
        }
        case LayerType::Division:
        {
            auto divisionQueueDescriptor = PolymorphicDowncast<const DivisionQueueDescriptor*>(&descriptor);
            return CreateDivision(*divisionQueueDescriptor, info);
        }
        case LayerType::ElementwiseBinary:
        {
            auto queueDescriptor = PolymorphicDowncast<const ElementwiseBinaryQueueDescriptor*>(&descriptor);
            return CreateWorkload(LayerType::ElementwiseBinary, *queueDescriptor, info);
        }
        case LayerType::ElementwiseUnary:
        {
            auto elementwiseUnaryQueueDescriptor
                    = PolymorphicDowncast<const ElementwiseUnaryQueueDescriptor*>(&descriptor);
            return CreateElementwiseUnary(*elementwiseUnaryQueueDescriptor, info);

        }
        case LayerType::FakeQuantization:
        {
            auto fakeQuantizationQueueDescriptor
                    = PolymorphicDowncast<const FakeQuantizationQueueDescriptor*>(&descriptor);
            return CreateFakeQuantization(*fakeQuantizationQueueDescriptor, info);
        }
        case LayerType::Fill:
        {
            auto fillQueueDescriptor = PolymorphicDowncast<const FillQueueDescriptor*>(&descriptor);
            return CreateFill(*fillQueueDescriptor, info);
        }
        case LayerType::Floor:
        {
            auto floorQueueDescriptor = PolymorphicDowncast<const FloorQueueDescriptor*>(&descriptor);
            return CreateFloor(*floorQueueDescriptor, info);
        }
        case LayerType::FullyConnected:
        {
            auto fullyConnectedQueueDescriptor
                    = PolymorphicDowncast<const FullyConnectedQueueDescriptor*>(&descriptor);
            return CreateFullyConnected(*fullyConnectedQueueDescriptor, info);
        }
        case LayerType::Gather:
        {
            auto gatherQueueDescriptor = PolymorphicDowncast<const GatherQueueDescriptor*>(&descriptor);
            return CreateGather(*gatherQueueDescriptor, info);
        }
        case LayerType::Input:
        {
            auto inputQueueDescriptor = PolymorphicDowncast<const InputQueueDescriptor*>(&descriptor);
            return CreateInput(*inputQueueDescriptor, info);
        }
        case LayerType::InstanceNormalization:
        {
            auto instanceNormalizationQueueDescriptor
                    = PolymorphicDowncast<const InstanceNormalizationQueueDescriptor*>(&descriptor);
            return CreateInstanceNormalization(*instanceNormalizationQueueDescriptor, info);
        }
        case LayerType::L2Normalization:
        {
            auto l2NormalizationQueueDescriptor
                    = PolymorphicDowncast<const L2NormalizationQueueDescriptor*>(&descriptor);
            return CreateL2Normalization(*l2NormalizationQueueDescriptor, info);
        }
        case LayerType::LogicalBinary:
        {
            auto logicalBinaryQueueDescriptor = PolymorphicDowncast<const LogicalBinaryQueueDescriptor*>(&descriptor);
            return CreateLogicalBinary(*logicalBinaryQueueDescriptor, info);
        }
        case LayerType::LogSoftmax:
        {
            auto logSoftmaxQueueDescriptor = PolymorphicDowncast<const LogSoftmaxQueueDescriptor*>(&descriptor);
            return CreateLogSoftmax(*logSoftmaxQueueDescriptor, info);
        }
        case LayerType::Lstm:
        {
            auto lstmQueueDescriptor = PolymorphicDowncast<const LstmQueueDescriptor*>(&descriptor);
            return CreateLstm(*lstmQueueDescriptor, info);
        }
        case LayerType::Maximum:
        {
            auto maximumQueueDescriptor = PolymorphicDowncast<const MaximumQueueDescriptor*>(&descriptor);
            return CreateMaximum(*maximumQueueDescriptor, info);
        }
        case LayerType::Mean:
        {
            auto meanQueueDescriptor = PolymorphicDowncast<const MeanQueueDescriptor*>(&descriptor);
            return CreateMean(*meanQueueDescriptor, info);
        }
        case LayerType::MemCopy:
        {
            auto memCopyQueueDescriptor = PolymorphicDowncast<const MemCopyQueueDescriptor*>(&descriptor);
            return CreateMemCopy(*memCopyQueueDescriptor, info);
        }
        case LayerType::MemImport:
        {
            auto memImportQueueDescriptor = PolymorphicDowncast<const MemImportQueueDescriptor*>(&descriptor);
            return CreateMemImport(*memImportQueueDescriptor, info);
        }
        case LayerType::Minimum:
        {
            auto minimumQueueDescriptor = PolymorphicDowncast<const MinimumQueueDescriptor*>(&descriptor);
            return CreateMinimum(*minimumQueueDescriptor, info);
        }
        case LayerType::Multiplication:
        {
            auto multiplicationQueueDescriptor
                    = PolymorphicDowncast<const MultiplicationQueueDescriptor*>(&descriptor);
            return CreateMultiplication(*multiplicationQueueDescriptor, info);
        }
        case LayerType::Normalization:
        {
            auto normalizationQueueDescriptor = PolymorphicDowncast<const NormalizationQueueDescriptor*>(&descriptor);
            return CreateNormalization(*normalizationQueueDescriptor, info);
        }
        case LayerType::Output:
        {
            auto outputQueueDescriptor = PolymorphicDowncast<const OutputQueueDescriptor*>(&descriptor);
            return CreateOutput(*outputQueueDescriptor, info);
        }
        case LayerType::Pad:
        {
            auto padQueueDescriptor = PolymorphicDowncast<const PadQueueDescriptor*>(&descriptor);
            return CreatePad(*padQueueDescriptor, info);
        }
        case LayerType::Permute:
        {
            auto permuteQueueDescriptor = PolymorphicDowncast<const PermuteQueueDescriptor*>(&descriptor);
            return CreatePermute(*permuteQueueDescriptor, info);
        }
        case LayerType::Pooling2d:
        {
            auto pooling2dQueueDescriptor = PolymorphicDowncast<const Pooling2dQueueDescriptor*>(&descriptor);
            return CreatePooling2d(*pooling2dQueueDescriptor, info);
        }
        case LayerType::Pooling3d:
        {
            auto pooling3dQueueDescriptor = PolymorphicDowncast<const Pooling3dQueueDescriptor*>(&descriptor);
            return CreatePooling3d(*pooling3dQueueDescriptor, info);
        }
        case LayerType::PreCompiled:
        {
            auto preCompiledQueueDescriptor = PolymorphicDowncast<const PreCompiledQueueDescriptor*>(&descriptor);
            return CreatePreCompiled(*preCompiledQueueDescriptor, info);
        }
        case LayerType::Prelu:
        {
            auto preluQueueDescriptor = PolymorphicDowncast<const PreluQueueDescriptor*>(&descriptor);
            return CreatePrelu(*preluQueueDescriptor, info);
        }
        case LayerType::QLstm:
        {
            auto qlstmQueueDescriptor = PolymorphicDowncast<const QLstmQueueDescriptor*>(&descriptor);
            return CreateQLstm(*qlstmQueueDescriptor, info);
        }
        case LayerType::Quantize:
        {
            auto quantizeQueueDescriptor = PolymorphicDowncast<const QuantizeQueueDescriptor*>(&descriptor);
            return CreateQuantize(*quantizeQueueDescriptor, info);
        }
        case LayerType::Rank:
        {
            auto rankQueueDescriptor = PolymorphicDowncast<const RankQueueDescriptor*>(&descriptor);
            return CreateRank(*rankQueueDescriptor, info);
        }
        case LayerType::Reduce:
        {
            auto reduceQueueDescriptor = PolymorphicDowncast<const ReduceQueueDescriptor*>(&descriptor);
            return CreateReduce(*reduceQueueDescriptor, info);
        }
        case LayerType::Reshape:
        {
            auto reshapeQueueDescriptor = PolymorphicDowncast<const ReshapeQueueDescriptor*>(&descriptor);
            return CreateReshape(*reshapeQueueDescriptor, info);
        }
        case LayerType::Resize:
        {
            auto resizeQueueDescriptor = PolymorphicDowncast<const ResizeQueueDescriptor*>(&descriptor);
            return CreateResize(*resizeQueueDescriptor, info);
        }
        case LayerType::Shape:
        {
            auto shapeQueueDescriptor = PolymorphicDowncast<const ShapeQueueDescriptor*>(&descriptor);
            return CreateShape(*shapeQueueDescriptor, info);
        }
        case LayerType::Slice:
        {
            auto sliceQueueDescriptor = PolymorphicDowncast<const SliceQueueDescriptor*>(&descriptor);
            return CreateSlice(*sliceQueueDescriptor, info);
        }
        case LayerType::Softmax:
        {
            auto softmaxQueueDescriptor = PolymorphicDowncast<const SoftmaxQueueDescriptor*>(&descriptor);
            return CreateSoftmax(*softmaxQueueDescriptor, info);
        }
        case LayerType::SpaceToBatchNd:
        {
            auto spaceToBatchNdQueueDescriptor
                    = PolymorphicDowncast<const SpaceToBatchNdQueueDescriptor*>(&descriptor);
            return CreateSpaceToBatchNd(*spaceToBatchNdQueueDescriptor, info);
        }
        case LayerType::SpaceToDepth:
        {
            auto spaceToDepthQueueDescriptor = PolymorphicDowncast<const SpaceToDepthQueueDescriptor*>(&descriptor);
            return CreateSpaceToDepth(*spaceToDepthQueueDescriptor, info);
        }
        case LayerType::Splitter:
        {
            auto splitterQueueDescriptor = PolymorphicDowncast<const SplitterQueueDescriptor*>(&descriptor);
            return CreateSplitter(*splitterQueueDescriptor, info);
        }
        case LayerType::Stack:
        {
            auto stackQueueDescriptor = PolymorphicDowncast<const StackQueueDescriptor*>(&descriptor);
            return CreateStack(*stackQueueDescriptor, info);
        }
        case LayerType::StridedSlice:
        {
            auto stridedSliceQueueDescriptor = PolymorphicDowncast<const StridedSliceQueueDescriptor*>(&descriptor);
            return CreateStridedSlice(*stridedSliceQueueDescriptor, info);
        }
        case LayerType::Subtraction:
        {
            auto subtractionQueueDescriptor = PolymorphicDowncast<const SubtractionQueueDescriptor*>(&descriptor);
            return CreateSubtraction(*subtractionQueueDescriptor, info);
        }
        case LayerType::Transpose:
        {
            auto transposeQueueDescriptor = PolymorphicDowncast<const TransposeQueueDescriptor*>(&descriptor);
            return CreateTranspose(*transposeQueueDescriptor, info);
        }
        case LayerType::TransposeConvolution2d:
        {
            auto transposeConvolution2dQueueDescriptor
                    = PolymorphicDowncast<const TransposeConvolution2dQueueDescriptor*>(&descriptor);
            return CreateTransposeConvolution2d(*transposeConvolution2dQueueDescriptor, info);
        }
        case LayerType::UnidirectionalSequenceLstm:
        {
            auto unidirectionalSequenceLstmQueueDescriptor
                    = PolymorphicDowncast<const UnidirectionalSequenceLstmQueueDescriptor*>(&descriptor);
            return CreateUnidirectionalSequenceLstm(*unidirectionalSequenceLstmQueueDescriptor, info);
        }
        default:
            return nullptr;
    }
}
ARMNN_NO_DEPRECATE_WARN_END

std::unique_ptr<IWorkload> IWorkloadFactory::CreateActivation(const ActivationQueueDescriptor& /*descriptor*/,
                                                              const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& /*descriptor*/,
                                                            const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateArgMinMax(const ArgMinMaxQueueDescriptor& /*descriptor*/,
                                                             const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateBatchNormalization(
    const BatchNormalizationQueueDescriptor& /*descriptor*/, const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateBatchToSpaceNd(const BatchToSpaceNdQueueDescriptor& /*desc*/,
                                                                  const WorkloadInfo& /*Info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateCast(const CastQueueDescriptor& /*descriptor*/,
                                                       const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateChannelShuffle(const ChannelShuffleQueueDescriptor& /*descriptor*/,
                                                                  const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateComparison(const ComparisonQueueDescriptor& /*descriptor*/,
                                                              const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateConcat(const ConcatQueueDescriptor& /*descriptor*/,
                                                          const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateConstant(const ConstantQueueDescriptor& /*descriptor*/,
                                                            const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateConvertFp16ToFp32(const ConvertFp16ToFp32QueueDescriptor& /*desc*/,
                                                                     const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateConvertFp32ToFp16(const ConvertFp32ToFp16QueueDescriptor& /*desc*/,
                                                                     const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateConvolution2d(const Convolution2dQueueDescriptor& /*descriptor*/,
                                                                 const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateConvolution3d(const Convolution3dQueueDescriptor& /*descriptor*/,
                                                                 const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateDebug(const DebugQueueDescriptor& /*descriptor*/,
                                                         const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateDepthToSpace(const DepthToSpaceQueueDescriptor& /*descriptor*/,
                                                                const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateDepthwiseConvolution2d(
    const DepthwiseConvolution2dQueueDescriptor& /*descriptor*/, const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateDequantize(
    const DequantizeQueueDescriptor& /*descriptor*/, const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateDetectionPostProcess(
    const DetectionPostProcessQueueDescriptor& /*descriptor*/, const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateDivision(const DivisionQueueDescriptor& /*descriptor*/,
                                                            const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateElementwiseUnary(const ElementwiseUnaryQueueDescriptor& /*desc*/,
                                                                    const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateFakeQuantization(const FakeQuantizationQueueDescriptor& /*desc*/,
                                                                    const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateFill(const FillQueueDescriptor& /*descriptor*/,
                                                        const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateFloor(const FloorQueueDescriptor& /*descriptor*/,
                                                         const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateFullyConnected(const FullyConnectedQueueDescriptor& /*descriptor*/,
                                                                  const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateGather(const GatherQueueDescriptor& /*descriptor*/,
                                                          const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateInstanceNormalization(
    const InstanceNormalizationQueueDescriptor& /*descriptor*/,
    const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateL2Normalization(const L2NormalizationQueueDescriptor& /*desc*/,
                                                                   const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateLogicalBinary(const LogicalBinaryQueueDescriptor& /*desc*/,
                                                                 const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateLogicalUnary(const ElementwiseUnaryQueueDescriptor& /*desc*/,
                                                                const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateLogSoftmax(const LogSoftmaxQueueDescriptor& /*descriptor*/,
                                                              const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateLstm(const LstmQueueDescriptor& /*descriptor*/,
                                                        const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateMaximum(const MaximumQueueDescriptor& /*descriptor*/,
                                                           const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateMean(const MeanQueueDescriptor& /*descriptor*/,
                                                        const WorkloadInfo& /*Info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateMemCopy(const MemCopyQueueDescriptor& /*descriptor*/,
                                                           const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateMemImport(const MemImportQueueDescriptor& /*descriptor*/,
                                                             const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateMerge(const MergeQueueDescriptor& /*descriptor*/,
                                                         const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateMinimum(const MinimumQueueDescriptor& /*descriptor*/,
                                                           const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateMultiplication(const MultiplicationQueueDescriptor& /*descriptor*/,
                                                                  const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateNormalization(const NormalizationQueueDescriptor& /*descriptor*/,
                                                                 const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateOutput(const OutputQueueDescriptor& /*descriptor*/,
                                                          const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreatePad(const PadQueueDescriptor& /*descriptor*/,
                                                       const WorkloadInfo& /*Info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreatePermute(const PermuteQueueDescriptor& /*descriptor*/,
                                                           const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreatePooling2d(const Pooling2dQueueDescriptor& /*descriptor*/,
                                                             const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreatePooling3d(const Pooling3dQueueDescriptor& /*descriptor*/,
                                                             const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreatePreCompiled(const PreCompiledQueueDescriptor& /*descriptor*/,
                                                               const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreatePrelu(const PreluQueueDescriptor &/*descriptor*/,
                                                         const WorkloadInfo &/*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateQuantize(const QuantizeQueueDescriptor& /*descriptor*/,
                                                            const WorkloadInfo& /*Info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateQLstm(const QLstmQueueDescriptor& /*descriptor*/,
                                                         const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateQuantizedLstm(const QuantizedLstmQueueDescriptor& /*descriptor*/,
                                                                 const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}
std::unique_ptr<IWorkload> IWorkloadFactory::CreateRank(const RankQueueDescriptor& /*descriptor*/,
                                                        const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateReduce(const ReduceQueueDescriptor& /*descriptor*/,
                                                          const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateReshape(const ReshapeQueueDescriptor& /*descriptor*/,
                                                           const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateResize(const ResizeQueueDescriptor& /*descriptor*/,
                                                            const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateShape(const ShapeQueueDescriptor& /*descriptor*/,
                                                         const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateSlice(const SliceQueueDescriptor& /*descriptor*/,
                                                         const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateSoftmax(const SoftmaxQueueDescriptor& /*descriptor*/,
                                                           const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateSplitter(const SplitterQueueDescriptor& /*descriptor*/,
                                                            const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateSpaceToBatchNd(const SpaceToBatchNdQueueDescriptor& /*descriptor*/,
                                                                  const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateSpaceToDepth(const SpaceToDepthQueueDescriptor& /*descriptor*/,
                                                                const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateStack(const StackQueueDescriptor& /*descriptor*/,
                                                         const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateStridedSlice(const StridedSliceQueueDescriptor& /*descriptor*/,
                                                                const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateSubtraction(const SubtractionQueueDescriptor& /*descriptor*/,
                                                               const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateSwitch(const SwitchQueueDescriptor& /*descriptor*/,
                                                          const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateTranspose(const TransposeQueueDescriptor& /*descriptor*/,
                                                             const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateTransposeConvolution2d(
    const TransposeConvolution2dQueueDescriptor& /*descriptor*/,
    const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateUnidirectionalSequenceLstm(
        const UnidirectionalSequenceLstmQueueDescriptor& /*descriptor*/,
        const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

std::unique_ptr<IWorkload> IWorkloadFactory::CreateInput(
        const InputQueueDescriptor& /*descriptor*/,
        const WorkloadInfo& /*info*/) const
{
    return std::unique_ptr<IWorkload>();
}

} // namepsace armnn
