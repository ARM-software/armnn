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

bool LayerSupportHandle::IsActivationSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const ActivationDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsActivationSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsAdditionSupported(const TensorInfo& input0,
                                             const TensorInfo& input1,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsAdditionSupported(input0, input1, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsArgMinMaxSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const ArgMinMaxDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsArgMinMaxSupported(input, output, descriptor, reasonIfUnsupported.value());
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
    return m_LayerSupport->IsBatchNormalizationSupported(input,
                                                         output,
                                                         mean,
                                                         var,
                                                         beta,
                                                         gamma,
                                                         descriptor,
                                                         reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsBatchToSpaceNdSupported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const BatchToSpaceNdDescriptor& descriptor,
                                                   Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsBatchToSpaceNdSupported(input,
                                                     output,
                                                     descriptor,
                                                     reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsCastSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsCastSupported(input, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsChannelShuffleSupported(const TensorInfo &input, const TensorInfo &output,
                                                   const ChannelShuffleDescriptor &descriptor,
                                                   Optional<std::string &> reasonIfUnsupported)
{
    return m_LayerSupport->IsChannelShuffleSupported(input,
                                                     output,
                                                     descriptor,
                                                     reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsComparisonSupported(const TensorInfo& input0,
                                               const TensorInfo& input1,
                                               const TensorInfo& output,
                                               const ComparisonDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsComparisonSupported(input0, input1, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsConcatSupported(const std::vector<const TensorInfo*> inputs,
                                           const TensorInfo& output,
                                           const OriginsDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsConcatSupported(inputs, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsConstantSupported(const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsConstantSupported(output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsConvertBf16ToFp32Supported(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsConvertBf16ToFp32Supported(input, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsConvertFp32ToBf16Supported(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsConvertFp32ToBf16Supported(input, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsConvertFp16ToFp32Supported(input, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsConvertFp32ToFp16Supported(input, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsConvolution2dSupported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const Convolution2dDescriptor& descriptor,
                                                  const TensorInfo& weights,
                                                  const Optional<TensorInfo>& biases,
                                                  Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsConvolution2dSupported(input,
                                                    output,
                                                    descriptor,
                                                    weights,
                                                    biases,
                                                    reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsConvolution3dSupported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const Convolution3dDescriptor& descriptor,
                                                  const TensorInfo& weights,
                                                  const Optional<TensorInfo>& biases,
                                                  Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsConvolution3dSupported(input,
                                                    output,
                                                    descriptor,
                                                    weights,
                                                    biases,
                                                    reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsDebugSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsDebugSupported(input, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsDepthToSpaceSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const DepthToSpaceDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsDepthToSpaceSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsDepthwiseConvolutionSupported(
        const TensorInfo& input,
        const TensorInfo& output,
        const DepthwiseConvolution2dDescriptor& descriptor,
        const TensorInfo& weights,
        const Optional<TensorInfo>& biases,
        Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsDepthwiseConvolutionSupported(input,
                                                           output,
                                                           descriptor,
                                                           weights,
                                                           biases,
                                                           reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsDequantizeSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsDequantizeSupported(input, output, reasonIfUnsupported.value());
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
    return m_LayerSupport->IsDetectionPostProcessSupported(boxEncodings,
                                                           scores,
                                                           anchors,
                                                           detectionBoxes,
                                                           detectionClasses,
                                                           detectionScores,
                                                           numDetections,
                                                           descriptor,
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
    return m_LayerSupport->IsDilatedDepthwiseConvolutionSupported(input,
                                                                  output,
                                                                  descriptor,
                                                                  weights,
                                                                  biases,
                                                                  reasonIfUnsupported);
}

bool LayerSupportHandle::IsDivisionSupported(const TensorInfo& input0,
                                             const TensorInfo& input1,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsDivisionSupported(input0, input1, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsElementwiseUnarySupported(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const ElementwiseUnaryDescriptor& descriptor,
                                                     Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsElementwiseUnarySupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsFakeQuantizationSupported(const TensorInfo& input,
                                                     const FakeQuantizationDescriptor& descriptor,
                                                     Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsFakeQuantizationSupported(input, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsFillSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const FillDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsFillSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsFloorSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsFloorSupported(input, output, reasonIfUnsupported.value());
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
                reasonIfUnsupported.value() =
                        "This backend might not support non constant weights. "
                        "If weights are constant make sure to set IsConstant when creating TensorInfo";
                return false;
            }
            if(descriptor.m_BiasEnabled)
            {
                if(!biases.IsConstant())
                {
                    reasonIfUnsupported.value() =
                            "This backend might not support non constant bias. "
                            "If bias are constant make sure to set IsConstant when creating TensorInfo";
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
            auto capability = GetCapability("NonConstWeights", m_BackendId);
            if (capability.has_value() && capability.value().GetValue().AsBool() == true)
            {
                return true;
            }
            return false;
        }
    }

    return m_LayerSupport->IsFullyConnectedSupported(input,
                                                     output,
                                                     weights,
                                                     biases,
                                                     descriptor,
                                                     reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsGatherSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output,
                                           const GatherDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsGatherSupported(input0, input1, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsInputSupported(const TensorInfo& input,
                                          Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsInputSupported(input, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsInstanceNormalizationSupported(
        const TensorInfo& input,
        const TensorInfo& output,
        const InstanceNormalizationDescriptor& descriptor,
        Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsInstanceNormalizationSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsL2NormalizationSupported(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    const L2NormalizationDescriptor& descriptor,
                                                    Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsL2NormalizationSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsLogicalBinarySupported(const TensorInfo& input0,
                                                  const TensorInfo& input1,
                                                  const TensorInfo& output,
                                                  const LogicalBinaryDescriptor& descriptor,
                                                  Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsLogicalBinarySupported(input0,
                                                    input1,
                                                    output,
                                                    descriptor,
                                                    reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsLogicalUnarySupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const ElementwiseUnaryDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsLogicalUnarySupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsLogSoftmaxSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const LogSoftmaxDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsLogSoftmaxSupported(input, output, descriptor, reasonIfUnsupported.value());
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
    return m_LayerSupport->IsLstmSupported(input,
                                           outputStateIn,
                                           cellStateIn,
                                           scratchBuffer,
                                           outputStateOut,
                                           cellStateOut,
                                           output,
                                           descriptor,
                                           paramsInfo,
                                           reasonIfUnsupported);
}

bool LayerSupportHandle::IsMaximumSupported(const TensorInfo& input0,
                                            const TensorInfo& input1,
                                            const TensorInfo& output,
                                            Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsMaximumSupported(input0, input1, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsMeanSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const MeanDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsMeanSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsMemCopySupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsMemCopySupported(input, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsMemImportSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsMemImportSupported(input, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsMergeSupported(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsMergeSupported(input0, input1, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsMinimumSupported(const TensorInfo& input0,
                                            const TensorInfo& input1,
                                            const TensorInfo& output,
                                            Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsMinimumSupported(input0, input1, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsMultiplicationSupported(const TensorInfo& input0,
                                                   const TensorInfo& input1,
                                                   const TensorInfo& output,
                                                   Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsMultiplicationSupported(input0, input1, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsNormalizationSupported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const NormalizationDescriptor& descriptor,
                                                  Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsNormalizationSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsOutputSupported(const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsOutputSupported(output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsPadSupported(const TensorInfo& input,
                                        const TensorInfo& output,
                                        const PadDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsPadSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsPermuteSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const PermuteDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsPermuteSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsPooling2dSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const Pooling2dDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsPooling2dSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsPreCompiledSupported(const TensorInfo& input,
                                                const PreCompiledDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsPreCompiledSupported(input, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsPreluSupported(const TensorInfo& input,
                                          const TensorInfo& alpha,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsPreluSupported(input, alpha, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsQuantizeSupported(const TensorInfo& input,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsQuantizeSupported(input, output, reasonIfUnsupported.value());
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
    return m_LayerSupport->IsQLstmSupported(input,
                                            previousOutputIn,
                                            previousCellStateIn,
                                            outputStateOut,
                                            cellStateOut,
                                            output,
                                            descriptor,
                                            paramsInfo,
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
    return m_LayerSupport->IsQuantizedLstmSupported(input,
                                                    previousCellStateIn,
                                                    previousOutputIn,
                                                    cellStateOut,
                                                    output,
                                                    paramsInfo,
                                                    reasonIfUnsupported);
}

bool LayerSupportHandle::IsRankSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsRankSupported(input, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsReduceSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const ReduceDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsReduceSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsReshapeSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const ReshapeDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsReshapeSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsResizeSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const ResizeDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsResizeSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsShapeSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsShapeSupported(input, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsSliceSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const SliceDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsSliceSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsSoftmaxSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const SoftmaxDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsSoftmaxSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsSpaceToBatchNdSupported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const SpaceToBatchNdDescriptor& descriptor,
                                                   Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsSpaceToBatchNdSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsSpaceToDepthSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const SpaceToDepthDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsSpaceToDepthSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsSplitterSupported(const TensorInfo& input,
                                             const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                                             const ViewsDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsSplitterSupported(input, outputs, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsStackSupported(const std::vector<const TensorInfo*>& inputs,
                                          const TensorInfo& output,
                                          const StackDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsStackSupported(inputs, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsStandInSupported(const std::vector<const TensorInfo*>& inputs,
                                            const std::vector<const TensorInfo*>& outputs,
                                            const StandInDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsStandInSupported(inputs, outputs, descriptor, reasonIfUnsupported.value());
}


bool LayerSupportHandle::IsStridedSliceSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const StridedSliceDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsStridedSliceSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsSubtractionSupported(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output,
                                                Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsSubtractionSupported(input0, input1, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsSwitchSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output0,
                                           const TensorInfo& output1,
                                           Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsSwitchSupported(input0, input1, output0, output1, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsTransposeConvolution2dSupported(
        const TensorInfo& input,
        const TensorInfo& output,
        const TransposeConvolution2dDescriptor& descriptor,
        const TensorInfo& weights,
        const Optional<TensorInfo>& biases,
        Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsTransposeConvolution2dSupported(input,
                                                             output,
                                                             descriptor,
                                                             weights,
                                                             biases,
                                                             reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsTransposeSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const TransposeDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsTransposeSupported(input, output, descriptor, reasonIfUnsupported.value());
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
    return m_LayerSupport->IsUnidirectionalSequenceLstmSupported(input,
                                                                 outputStateIn,
                                                                 cellStateIn,
                                                                 output,
                                                                 hiddenStateOutput,
                                                                 cellStateOutput,
                                                                 descriptor,
                                                                 paramsInfo,
                                                                 reasonIfUnsupported);
}

}