//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefLayerSupport.hpp"
#include "RefBackendId.hpp"

#include <InternalTypes.hpp>
#include <LayerSupportCommon.hpp>
#include <armnn/Types.hpp>

#include <backendsCommon/BackendRegistry.hpp>

#include <boost/core/ignore_unused.hpp>

using namespace boost;

namespace armnn
{

namespace
{

template<typename Float32Func, typename Uint8Func, typename ... Params>
bool IsSupportedForDataTypeRef(Optional<std::string&> reasonIfUnsupported,
                               DataType dataType,
                               Float32Func floatFuncPtr,
                               Uint8Func uint8FuncPtr,
                               Params&&... params)
{
    return IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                         dataType,
                                         &FalseFunc<Params...>,
                                         floatFuncPtr,
                                         uint8FuncPtr,
                                         std::forward<Params>(params)...);
}

} // anonymous namespace

bool RefLayerSupport::IsActivationSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const ActivationDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsAdditionSupported(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsBatchNormalizationSupported(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    const TensorInfo& mean,
                                                    const TensorInfo& var,
                                                    const TensorInfo& beta,
                                                    const TensorInfo& gamma,
                                                    const BatchNormalizationDescriptor& descriptor,
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    ignore_unused(mean);
    ignore_unused(var);
    ignore_unused(beta);
    ignore_unused(gamma);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsBatchToSpaceNdSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const BatchToSpaceNdDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(descriptor);
    return (IsSupportedForDataTypeRef(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>) &&
            IsSupportedForDataTypeRef(reasonIfUnsupported,
                                      output.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>));
}

bool RefLayerSupport::IsConstantSupported(const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     output.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    return (IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                          input.GetDataType(),
                                          &TrueFunc<>,
                                          &FalseInputFuncF32<>,
                                          &FalseFuncU8<>) &&
            IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                          output.GetDataType(),
                                          &FalseOutputFuncF16<>,
                                          &TrueFunc<>,
                                          &FalseFuncU8<>));
}

bool RefLayerSupport::IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    return (IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                          input.GetDataType(),
                                          &FalseInputFuncF16<>,
                                          &TrueFunc<>,
                                          &FalseFuncU8<>) &&
            IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                          output.GetDataType(),
                                          &TrueFunc<>,
                                          &FalseOutputFuncF32<>,
                                          &FalseFuncU8<>));
}

bool RefLayerSupport::IsConvolution2dSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const Convolution2dDescriptor& descriptor,
                                               const TensorInfo& weights,
                                               const Optional<TensorInfo>& biases,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(weights);
    ignore_unused(biases);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsDebugSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const DebugDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const DepthwiseConvolution2dDescriptor& descriptor,
                                                      const TensorInfo& weights,
                                                      const Optional<TensorInfo>& biases,
                                                      Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(weights);
    ignore_unused(biases);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsDivisionSupported(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsEqualSupported(const TensorInfo& input0,
                                       const TensorInfo& input1,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input0);
    ignore_unused(input1);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsFakeQuantizationSupported(const TensorInfo& input,
                                                  const FakeQuantizationDescriptor& descriptor,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &FalseFuncU8<>);
}

bool RefLayerSupport::IsFloorSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &FalseFuncU8<>);
}

bool RefLayerSupport::IsFullyConnectedSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const TensorInfo& weights,
                                                const TensorInfo& biases,
                                                const FullyConnectedDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    ignore_unused(weights);
    ignore_unused(biases);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsInputSupported(const TensorInfo& input,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsL2NormalizationSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const L2NormalizationDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &FalseFuncU8<>);
}

bool RefLayerSupport::IsLstmSupported(const TensorInfo& input,
                                      const TensorInfo& outputStateIn,
                                      const TensorInfo& cellStateIn,
                                      const TensorInfo& scratchBuffer,
                                      const TensorInfo& outputStateOut,
                                      const TensorInfo& cellStateOut,
                                      const TensorInfo& output,
                                      const LstmDescriptor& descriptor,
                                      const TensorInfo& inputToForgetWeights,
                                      const TensorInfo& inputToCellWeights,
                                      const TensorInfo& inputToOutputWeights,
                                      const TensorInfo& recurrentToForgetWeights,
                                      const TensorInfo& recurrentToCellWeights,
                                      const TensorInfo& recurrentToOutputWeights,
                                      const TensorInfo& forgetGateBias,
                                      const TensorInfo& cellBias,
                                      const TensorInfo& outputGateBias,
                                      const TensorInfo* inputToInputWeights,
                                      const TensorInfo* recurrentToInputWeights,
                                      const TensorInfo* cellToInputWeights,
                                      const TensorInfo* inputGateBias,
                                      const TensorInfo* projectionWeights,
                                      const TensorInfo* projectionBias,
                                      const TensorInfo* cellToForgetWeights,
                                      const TensorInfo* cellToOutputWeights,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(outputStateIn);
    ignore_unused(cellStateIn);
    ignore_unused(scratchBuffer);
    ignore_unused(outputStateOut);
    ignore_unused(cellStateOut);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(inputToForgetWeights);
    ignore_unused(inputToCellWeights);
    ignore_unused(inputToOutputWeights);
    ignore_unused(recurrentToForgetWeights);
    ignore_unused(recurrentToCellWeights);
    ignore_unused(recurrentToOutputWeights);
    ignore_unused(forgetGateBias);
    ignore_unused(cellBias);
    ignore_unused(outputGateBias);
    ignore_unused(inputToInputWeights);
    ignore_unused(recurrentToInputWeights);
    ignore_unused(cellToInputWeights);
    ignore_unused(inputGateBias);
    ignore_unused(projectionWeights);
    ignore_unused(projectionBias);
    ignore_unused(cellToForgetWeights);
    ignore_unused(cellToOutputWeights);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &FalseFuncU8<>);
}

bool RefLayerSupport::IsMaximumSupported(const TensorInfo& input0,
                                         const TensorInfo& input1,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsMeanSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const MeanDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsMergerSupported(const std::vector<const TensorInfo*> inputs,
                                        const TensorInfo& output,
                                        const OriginsDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(descriptor);
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     inputs[0]->GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsMinimumSupported(const TensorInfo& input0,
                                         const TensorInfo& input1,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsMultiplicationSupported(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsNormalizationSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const NormalizationDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &FalseFuncU8<>);
}

bool RefLayerSupport::IsOutputSupported(const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     output.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsPadSupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const PadDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsPermuteSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const PermuteDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsPooling2dSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const Pooling2dDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsReshapeSupported(const TensorInfo& input,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsResizeBilinearSupported(const TensorInfo& input,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsSoftmaxSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const SoftmaxDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsSpaceToBatchNdSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const SpaceToBatchNdDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsSplitterSupported(const TensorInfo& input,
                                          const ViewsDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsStridedSliceSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const StridedSliceDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsSubtractionSupported(const TensorInfo& input0,
                                             const TensorInfo& input1,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

} // namespace armnn
