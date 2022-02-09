//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/BackendHelper.hpp>
#include <armnn/BackendRegistry.hpp>
#include <armnn/Logging.hpp>

#include <armnn/backends/IBackendInternal.hpp>

namespace armnn
{

// Return LayerSupportHandle instead of the previous pointer to ILayerSupport.
LayerSupportHandle GetILayerSupportByBackendId(const armnn::BackendId& backend)
{
    BackendRegistry& backendRegistry = armnn::BackendRegistryInstance();

    if (!backendRegistry.IsBackendRegistered(backend))
    {
        return LayerSupportHandle(nullptr);
    }

    auto factoryFunc = backendRegistry.GetFactory(backend);
    auto backendObject = factoryFunc();
    return LayerSupportHandle(backendObject->GetLayerSupport(), backend);
}

Optional<const BackendOptions::BackendOption> GetCapability(const std::string& backendCapabilityName,
                                                            const BackendCapabilities& capabilities)
{
    for (size_t i=0; i < capabilities.GetOptionCount(); i++)
    {
        const auto& capability = capabilities.GetOption(i);
        if (backendCapabilityName == capability.GetName())
        {
            return capability;
        }
    }
    return EmptyOptional();
}

Optional<const BackendOptions::BackendOption> GetCapability(const std::string& backendCapabilityName,
                                                            const armnn::BackendId& backend)
{
    auto const& backendRegistry = armnn::BackendRegistryInstance();
    if (backendRegistry.IsBackendRegistered(backend))
    {
        auto factoryFunc = backendRegistry.GetFactory(backend);
        auto backendObject = factoryFunc();
        auto capabilities = backendObject->GetCapabilities();
        return GetCapability(backendCapabilityName, capabilities);
    }
    return EmptyOptional();
}

bool HasCapability(const std::string& name, const BackendCapabilities& capabilities)
{
    return GetCapability(name, capabilities).has_value();
}

bool HasCapability(const std::string& name, const armnn::BackendId& backend)
{
    return GetCapability(name, backend).has_value();
}

bool HasCapability(const BackendOptions::BackendOption& capability, const BackendCapabilities& capabilities)
{
    for (size_t i=0; i < capabilities.GetOptionCount(); i++)
    {
        const auto& backendCapability = capabilities.GetOption(i);
        if (capability.GetName() == backendCapability.GetName())
        {
            if (capability.GetValue().IsBool() && backendCapability.GetValue().IsBool())
            {
                return capability.GetValue().AsBool() == backendCapability.GetValue().AsBool();
            }
            else if(capability.GetValue().IsFloat() && backendCapability.GetValue().IsFloat())
            {
                return capability.GetValue().AsFloat() == backendCapability.GetValue().AsFloat();
            }
            else if(capability.GetValue().IsInt() && backendCapability.GetValue().IsInt())
            {
                return capability.GetValue().AsInt() == backendCapability.GetValue().AsInt();
            }
            else if(capability.GetValue().IsString() && backendCapability.GetValue().IsString())
            {
                return capability.GetValue().AsString() == backendCapability.GetValue().AsString();
            }
            else if(capability.GetValue().IsUnsignedInt() && backendCapability.GetValue().IsUnsignedInt())
            {
                return capability.GetValue().AsUnsignedInt() == backendCapability.GetValue().AsUnsignedInt();
            }
        }
    }
    return false;
}

bool HasCapability(const BackendOptions::BackendOption& backendOption, const armnn::BackendId& backend)
{
    auto const& backendRegistry = armnn::BackendRegistryInstance();
    if (backendRegistry.IsBackendRegistered(backend))
    {
        auto factoryFunc = backendRegistry.GetFactory(backend);
        auto backendObject = factoryFunc();
        auto capabilities = backendObject->GetCapabilities();
        return HasCapability(backendOption, capabilities);
    }
    return false;
}

/// Convenience function to check a capability on a backend
bool IsCapabilitySupported(const armnn::BackendId& backend, armnn::BackendCapability capability)
{
    bool hasCapability = false;
    auto const& backendRegistry = armnn::BackendRegistryInstance();
    if (backendRegistry.IsBackendRegistered(backend))
    {
        auto factoryFunc = backendRegistry.GetFactory(backend);
        auto backendObject = factoryFunc();
        ARMNN_NO_DEPRECATE_WARN_BEGIN
        hasCapability = backendObject->HasCapability(capability);
        ARMNN_NO_DEPRECATE_WARN_END
    }
    return hasCapability;
}

unsigned int GetNumberOfCacheFiles(const armnn::BackendId& backend)
{
    auto const& backendRegistry = armnn::BackendRegistryInstance();
    if (backendRegistry.IsBackendRegistered(backend))
    {
        auto factoryFunc = backendRegistry.GetFactory(backend);
        auto backendObject = factoryFunc();
        return backendObject->GetNumberOfCacheFiles();
    }
    return 0;
}

bool LayerSupportHandle::IsBackendRegistered() const
{
    if (m_LayerSupport)
    {
        return true;
    }

    return false;
}

using TensorInfos = std::vector<TensorInfo>;

bool LayerSupportHandle::IsActivationSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const ActivationDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Activation,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsAdditionSupported(const TensorInfo& input0,
                                             const TensorInfo& input1,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input0, input1, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Addition,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsArgMinMaxSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const ArgMinMaxDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::ArgMinMax,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsBatchNormalizationSupported(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const TensorInfo& mean,
                                                       const TensorInfo& var,
                                                       const TensorInfo& beta,
                                                       const TensorInfo& gamma,
                                                       const BatchNormalizationDescriptor& descriptor,
                                                       Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output, mean, var, beta, gamma};

    return m_LayerSupport->IsLayerSupported(LayerType::BatchNormalization,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsBatchToSpaceNdSupported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const BatchToSpaceNdDescriptor& descriptor,
                                                   Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::BatchToSpaceNd,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsCastSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Cast,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsChannelShuffleSupported(const TensorInfo &input,
                                                   const TensorInfo &output,
                                                   const ChannelShuffleDescriptor &descriptor,
                                                   Optional<std::string &> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::ChannelShuffle,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsComparisonSupported(const TensorInfo& input0,
                                               const TensorInfo& input1,
                                               const TensorInfo& output,
                                               const ComparisonDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input0, input1, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Comparison,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsConcatSupported(const std::vector<const TensorInfo*> inputs,
                                           const TensorInfo& output,
                                           const OriginsDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos;
    for (const TensorInfo* inputInfo : inputs)
    {
        infos.push_back(*inputInfo);
    }
    infos.push_back(output);

    return m_LayerSupport->IsLayerSupported(LayerType::Concat,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsConstantSupported(const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{output};

    return m_LayerSupport->IsLayerSupported(LayerType::Constant,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsConvertBf16ToFp32Supported(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::ConvertBf16ToFp32,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsConvertFp32ToBf16Supported(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::ConvertFp32ToBf16,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::ConvertFp16ToFp32,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::ConvertFp32ToFp16,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsConvolution2dSupported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const Convolution2dDescriptor& descriptor,
                                                  const TensorInfo& weights,
                                                  const Optional<TensorInfo>& biases,
                                                  Optional<std::string&> reasonIfUnsupported)
{
    TensorInfo biasesVal =  biases.has_value() ? biases.value() : TensorInfo();
    TensorInfos infos{input, output, weights, biasesVal};

    return m_LayerSupport->IsLayerSupported(LayerType::Convolution2d,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsConvolution3dSupported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const Convolution3dDescriptor& descriptor,
                                                  const TensorInfo& weights,
                                                  const Optional<TensorInfo>& biases,
                                                  Optional<std::string&> reasonIfUnsupported)
{
    TensorInfo biasesVal =  biases.has_value() ? biases.value() : TensorInfo();
    TensorInfos infos{input, output, weights, biasesVal};

    return m_LayerSupport->IsLayerSupported(LayerType::Convolution3d,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsDebugSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Debug,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsDepthToSpaceSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const DepthToSpaceDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::DepthToSpace,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsDepthwiseConvolutionSupported(
        const TensorInfo& input,
        const TensorInfo& output,
        const DepthwiseConvolution2dDescriptor& descriptor,
        const TensorInfo& weights,
        const Optional<TensorInfo>& biases,
        Optional<std::string&> reasonIfUnsupported)
{
    TensorInfo biasesVal =  biases.has_value() ? biases.value() : TensorInfo();
    TensorInfos infos{input, output, weights, biasesVal};

    return m_LayerSupport->IsLayerSupported(LayerType::DepthwiseConvolution2d,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsDequantizeSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Dequantize,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsDetectionPostProcessSupported(const TensorInfo& boxEncodings,
                                                         const TensorInfo& scores,
                                                         const TensorInfo& anchors,
                                                         const TensorInfo& detectionBoxes,
                                                         const TensorInfo& detectionClasses,
                                                         const TensorInfo& detectionScores,
                                                         const TensorInfo& numDetections,
                                                         const DetectionPostProcessDescriptor& descriptor,
                                                         Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{boxEncodings, scores, anchors, detectionBoxes, detectionClasses, detectionScores, numDetections};

    return m_LayerSupport->IsLayerSupported(LayerType::DetectionPostProcess,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsDilatedDepthwiseConvolutionSupported(
        const TensorInfo& input,
        const TensorInfo& output,
        const DepthwiseConvolution2dDescriptor& descriptor,
        const TensorInfo& weights,
        const Optional<TensorInfo>& biases,
        Optional<std::string&> reasonIfUnsupported)
{
    TensorInfo biasesVal =  biases.has_value() ? biases.value() : TensorInfo();
    TensorInfos infos{input, output, weights, biasesVal};

    return m_LayerSupport->IsLayerSupported(LayerType::DepthwiseConvolution2d,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsDivisionSupported(const TensorInfo& input0,
                                             const TensorInfo& input1,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input0, input1, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Division,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsElementwiseUnarySupported(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const ElementwiseUnaryDescriptor& descriptor,
                                                     Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::ElementwiseUnary,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsFakeQuantizationSupported(const TensorInfo& input,
                                                     const FakeQuantizationDescriptor& descriptor,
                                                     Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input};

    return m_LayerSupport->IsLayerSupported(LayerType::FakeQuantization,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsFillSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const FillDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Fill,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsFloorSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Floor,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsFullyConnectedSupported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const TensorInfo& weights,
                                                   const TensorInfo& biases,
                                                   const FullyConnectedDescriptor& descriptor,
                                                   Optional<std::string&> reasonIfUnsupported)
{
    if(!m_BackendId.IsUndefined())
    {
        auto capability = GetCapability("ConstantTensorsAsInputs", m_BackendId);
        if(!capability.has_value() || capability.value().GetValue().AsBool() == false)
        {
            if(!weights.IsConstant())
            {
                if (reasonIfUnsupported.has_value())
                {
                    reasonIfUnsupported.value() =
                            "This backend might not support non constant weights. "
                            "If weights are constant make sure to set IsConstant when creating TensorInfo";
                }

                return false;
            }
            if(descriptor.m_BiasEnabled)
            {
                if(!biases.IsConstant())
                {
                    if (reasonIfUnsupported.has_value())
                    {
                        reasonIfUnsupported.value() =
                                "This backend might not support non constant weights. "
                                "If weights are constant make sure to set IsConstant when creating TensorInfo";
                    }
                    return false;
                }
            }

            // At the first stage we will only print a warning. this is to give
            // backend developers a chance to adopt and read weights from input slots.
            ARMNN_LOG(warning) << "The backend makes use of a deprecated interface to read constant tensors. "
                                  "If you are a backend developer please find more information in our "
                                  "doxygen documentation on github https://github.com/ARM-software/armnn "
                                  "under the keyword 'ConstTensorsAsInputs'.";
        }

        if(!descriptor.m_ConstantWeights)
        {
            capability = GetCapability("NonConstWeights", m_BackendId);
            if (capability.has_value() && capability.value().GetValue().AsBool() == true)
            {
                return true;
            }
            return false;
        }
    }

    TensorInfos infos{input, output, weights, biases};

    return m_LayerSupport->IsLayerSupported(LayerType::FullyConnected,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsGatherSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output,
                                           const GatherDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input0, input1, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Gather,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsInputSupported(const TensorInfo& input,
                                          Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input};

    return m_LayerSupport->IsLayerSupported(LayerType::Input,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsInstanceNormalizationSupported(
        const TensorInfo& input,
        const TensorInfo& output,
        const InstanceNormalizationDescriptor& descriptor,
        Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::InstanceNormalization,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsL2NormalizationSupported(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    const L2NormalizationDescriptor& descriptor,
                                                    Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::L2Normalization,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsLogicalBinarySupported(const TensorInfo& input0,
                                                  const TensorInfo& input1,
                                                  const TensorInfo& output,
                                                  const LogicalBinaryDescriptor& descriptor,
                                                  Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input0, input1, output};

    return m_LayerSupport->IsLayerSupported(LayerType::LogicalBinary,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsLogicalUnarySupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const ElementwiseUnaryDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::ElementwiseUnary,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsLogSoftmaxSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const LogSoftmaxDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::LogSoftmax,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsLstmSupported(const TensorInfo& input,
                                         const TensorInfo& outputStateIn,
                                         const TensorInfo& cellStateIn,
                                         const TensorInfo& scratchBuffer,
                                         const TensorInfo& outputStateOut,
                                         const TensorInfo& cellStateOut,
                                         const TensorInfo& output,
                                         const LstmDescriptor& descriptor,
                                         const LstmInputParamsInfo& paramsInfo,
                                         Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, outputStateIn, cellStateIn, scratchBuffer, outputStateOut, cellStateOut, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Lstm,
                                            infos,
                                            descriptor,
                                            paramsInfo,
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsMaximumSupported(const TensorInfo& input0,
                                            const TensorInfo& input1,
                                            const TensorInfo& output,
                                            Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input0, input1, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Maximum,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsMeanSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const MeanDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Mean,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsMemCopySupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::MemCopy,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsMemImportSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::MemImport,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsMergeSupported(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input0, input1, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Merge,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsMinimumSupported(const TensorInfo& input0,
                                            const TensorInfo& input1,
                                            const TensorInfo& output,
                                            Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input0, input1, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Minimum,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsMultiplicationSupported(const TensorInfo& input0,
                                                   const TensorInfo& input1,
                                                   const TensorInfo& output,
                                                   Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input0, input1, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Multiplication,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsNormalizationSupported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const NormalizationDescriptor& descriptor,
                                                  Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Normalization,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsOutputSupported(const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{output};

    return m_LayerSupport->IsLayerSupported(LayerType::Output,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsPadSupported(const TensorInfo& input,
                                        const TensorInfo& output,
                                        const PadDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Pad,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsPermuteSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const PermuteDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Permute,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsPooling2dSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const Pooling2dDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Pooling2d,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsPooling3dSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const Pooling3dDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Pooling3d,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsPreCompiledSupported(const TensorInfo& input,
                                                const PreCompiledDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input};

    return m_LayerSupport->IsLayerSupported(LayerType::PreCompiled,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsPreluSupported(const TensorInfo& input,
                                          const TensorInfo& alpha,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, alpha, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Prelu,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsQuantizeSupported(const TensorInfo& input,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Quantize,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsQLstmSupported(const TensorInfo& input,
                                          const TensorInfo& previousOutputIn,
                                          const TensorInfo& previousCellStateIn,
                                          const TensorInfo& outputStateOut,
                                          const TensorInfo& cellStateOut,
                                          const TensorInfo& output,
                                          const QLstmDescriptor& descriptor,
                                          const LstmInputParamsInfo& paramsInfo,
                                          Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, previousOutputIn, previousCellStateIn, outputStateOut, cellStateOut, output};

    return m_LayerSupport->IsLayerSupported(LayerType::QLstm,
                                            infos,
                                            descriptor,
                                            paramsInfo,
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsQuantizedLstmSupported(const TensorInfo& input,
                                                  const TensorInfo& previousCellStateIn,
                                                  const TensorInfo& previousOutputIn,
                                                  const TensorInfo& cellStateOut,
                                                  const TensorInfo& output,
                                                  const QuantizedLstmInputParamsInfo& paramsInfo,
                                                  Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, previousCellStateIn, previousOutputIn, cellStateOut, output};

    return m_LayerSupport->IsLayerSupported(LayerType::QuantizedLstm,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            paramsInfo,
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsRankSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Rank,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsReduceSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const ReduceDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Reduce,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsReshapeSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const ReshapeDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Reshape,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsResizeSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const ResizeDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Resize,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsShapeSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Shape,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsSliceSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const SliceDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Slice,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsSoftmaxSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const SoftmaxDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Softmax,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsSpaceToBatchNdSupported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const SpaceToBatchNdDescriptor& descriptor,
                                                   Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::SpaceToBatchNd,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsSpaceToDepthSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const SpaceToDepthDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::SpaceToDepth,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsSplitterSupported(const TensorInfo& input,
                                             const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                                             const ViewsDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input};
    for (TensorInfo outInfo : outputs)
    {
        infos.push_back(outInfo);
    }

    return m_LayerSupport->IsLayerSupported(LayerType::Splitter,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsStackSupported(const std::vector<const TensorInfo*>& inputs,
                                          const TensorInfo& output,
                                          const StackDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos;
    for (const TensorInfo* inputInfo : inputs)
    {
        infos.push_back(*inputInfo);
    }
    infos.push_back(output);

    return m_LayerSupport->IsLayerSupported(LayerType::Stack,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsStandInSupported(const std::vector<const TensorInfo*>& inputs,
                                            const std::vector<const TensorInfo*>& outputs,
                                            const StandInDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos;
    for (const TensorInfo* inputInfo : inputs)
    {
        infos.push_back(*inputInfo);
    }
    for (const TensorInfo* outputInfo : outputs)
    {
        infos.push_back(*outputInfo);
    }

    return m_LayerSupport->IsLayerSupported(LayerType::StandIn,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}


bool LayerSupportHandle::IsStridedSliceSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const StridedSliceDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::StridedSlice,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsSubtractionSupported(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output,
                                                Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input0, input1, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Subtraction,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsSwitchSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output0,
                                           const TensorInfo& output1,
                                           Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input0, input1, output0, output1};

    return m_LayerSupport->IsLayerSupported(LayerType::Switch,
                                            infos,
                                            BaseDescriptor(),
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsTransposeConvolution2dSupported(
        const TensorInfo& input,
        const TensorInfo& output,
        const TransposeConvolution2dDescriptor& descriptor,
        const TensorInfo& weights,
        const Optional<TensorInfo>& biases,
        Optional<std::string&> reasonIfUnsupported)
{
    TensorInfo biasesVal =  biases.has_value() ? biases.value() : TensorInfo();
    TensorInfos infos{input, output, weights, biasesVal};

    return m_LayerSupport->IsLayerSupported(LayerType::TransposeConvolution2d,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsTransposeSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const TransposeDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported)
{
    TensorInfos infos{input, output};

    return m_LayerSupport->IsLayerSupported(LayerType::Transpose,
                                            infos,
                                            descriptor,
                                            EmptyOptional(),
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

bool LayerSupportHandle::IsUnidirectionalSequenceLstmSupported(const TensorInfo& input,
                                                               const TensorInfo& outputStateIn,
                                                               const TensorInfo& cellStateIn,
                                                               const TensorInfo& output,
                                                               const Optional<TensorInfo>& hiddenStateOutput,
                                                               const Optional<TensorInfo>& cellStateOutput,
                                                               const LstmDescriptor& descriptor,
                                                               const LstmInputParamsInfo& paramsInfo,
                                                               Optional<std::string&> reasonIfUnsupported)
{
    TensorInfo hiddenStateOutputVal =  hiddenStateOutput.has_value() ? hiddenStateOutput.value() : TensorInfo();
    TensorInfo cellStateOutputVal   =  cellStateOutput.has_value() ? cellStateOutput.value() : TensorInfo();
    TensorInfos infos{input, outputStateIn, cellStateIn, hiddenStateOutputVal, cellStateOutputVal, output};

    return m_LayerSupport->IsLayerSupported(LayerType::UnidirectionalSequenceLstm,
                                            infos,
                                            descriptor,
                                            paramsInfo,
                                            EmptyOptional(),
                                            reasonIfUnsupported);
}

}