//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/BackendHelper.hpp>
#include <armnn/BackendRegistry.hpp>

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

/// Convenience function to check a capability on a backend
bool IsCapabilitySupported(const armnn::BackendId& backend, armnn::BackendCapability capability)
{
    bool hasCapability = false;
    auto const& backendRegistry = armnn::BackendRegistryInstance();
    if (backendRegistry.IsBackendRegistered(backend))
    {
        auto factoryFunc = backendRegistry.GetFactory(backend);
        auto backendObject = factoryFunc();
        hasCapability = backendObject->HasCapability(capability);
    }
    return hasCapability;
}

bool LayerSupportHandle::IsBackendRegistered() const
{
    if (m_LayerSupport)
    {
        return true;
    }

    return false;
}


bool LayerSupportHandle::IsAbsSupported(const TensorInfo& input,
                                        const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported)
{
    // Call the IsXXXLayerSupport function of the specific backend.
    return m_LayerSupport->IsAbsSupported(input, output, reasonIfUnsupported.value());
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

bool LayerSupportHandle::IsEqualSupported(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsEqualSupported(input0, input1, output, reasonIfUnsupported.value());
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
    if(!descriptor.m_ConstantWeights && !m_BackendId.IsUndefined())
    {
        bool result = false;
        result = IsCapabilitySupported(m_BackendId, BackendCapability::NonConstWeights);
        if (!result)
        {
            return result;
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
                                           Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsGatherSupported(input0, input1, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsGatherSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output,
                                           const GatherDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsGatherSupported(input0, input1, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsGreaterSupported(const TensorInfo& input0,
                                            const TensorInfo& input1,
                                            const TensorInfo& ouput,
                                            Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsGreaterSupported(input0, input1, ouput, reasonIfUnsupported.value());
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

bool LayerSupportHandle::IsMergerSupported(const std::vector<const TensorInfo*> inputs,
                                           const TensorInfo& output,
                                           const OriginsDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsMergerSupported(inputs, output, descriptor, reasonIfUnsupported.value());
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

bool LayerSupportHandle::IsResizeBilinearSupported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsResizeBilinearSupported(input, output, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsResizeSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const ResizeDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsResizeSupported(input, output, descriptor, reasonIfUnsupported.value());
}

bool LayerSupportHandle::IsRsqrtSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsRsqrtSupported(input, output, reasonIfUnsupported.value());
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
                                             const ViewsDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported)
{
    return m_LayerSupport->IsSplitterSupported(input, descriptor, reasonIfUnsupported.value());
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

}