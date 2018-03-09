//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "LayerSupportCommon.hpp"
#include "RefLayerSupport.hpp"
#include <armnn/Descriptors.hpp>
#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>

#include <boost/core/ignore_unused.hpp>

#include "InternalTypes.hpp"

using namespace boost;

namespace armnn
{

template<typename Float32Func, typename Uint8Func, typename ... Params>
bool IsSupportedForDataTypeRef(std::string* reasonIfUnsupported,
                               DataType dataType,
                               Float32Func floatFuncPtr,
                               Uint8Func uint8FuncPtr,
                               Params&&... params)
{
    return IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                         dataType,
                                         floatFuncPtr,
                                         uint8FuncPtr,
                                         std::forward<Params>(params)...);
}

bool IsActivationSupportedRef(const TensorInfo& input,
                              const ActivationDescriptor& descriptor,
                              std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsAdditionSupportedRef(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            std::string* reasonIfUnsupported)
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsBatchNormalizationSupportedRef(const TensorInfo& input,
                                      const BatchNormalizationDescriptor& descriptor,
                                      std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsConstantSupportedRef(const TensorInfo& output,
                            std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     output.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsConvolution2dSupportedRef(const TensorInfo& input,
                                 const Convolution2dDescriptor& descriptor,
                                 const TensorInfo& weights,
                                 std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsDepthwiseConvolutionSupportedRef(const TensorInfo& input,
                                        const DepthwiseConvolution2dDescriptor& descriptor,
                                        const TensorInfo& weights,
                                        std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    ignore_unused(weights);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsFullyConnectedSupportedRef(const TensorInfo& input,
                                  const FullyConnectedDescriptor& descriptor,
                                  std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsInputSupportedRef(const TensorInfo& input,
                         std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsL2NormalizationSupportedRef(const TensorInfo& input,
                                   std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &FalseFuncU8<>);
}

bool IsMergerSupportedRef(const std::vector<const TensorInfo*> inputs,
                          const OriginsDescriptor& descriptor,
                          std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     inputs[0]->GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsMultiplicationSupportedRef(const TensorInfo& input0,
                                  const TensorInfo& input1,
                                  std::string* reasonIfUnsupported)
{
    ignore_unused(input1);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsNormalizationSupportedRef(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const NormalizationDescriptor& descriptor,
                                 std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &FalseFuncU8<>);
}

bool IsOutputSupportedRef(const TensorInfo& output,
                          std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     output.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsPermuteSupportedRef(const TensorInfo& input,
                           const TensorInfo& output,
                           const PermuteDescriptor& descriptor,
                           std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsPooling2dSupportedRef(const TensorInfo& input,
                             const TensorInfo& output,
                             const Pooling2dDescriptor& descriptor,
                             std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsResizeBilinearSupportedRef(const TensorInfo& input,
                                  std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsSoftmaxSupportedRef(const TensorInfo& input,
                           const SoftmaxDescriptor& descriptor,
                           std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsSplitterSupportedRef(const TensorInfo& input,
                            const ViewsDescriptor& descriptor,
                            std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsFakeQuantizationSupportedRef(const TensorInfo& input,
                                    const FakeQuantizationDescriptor& descriptor,
                                    std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &FalseFuncU8<>);
}

bool IsReshapeSupportedRef(const TensorInfo& input,
                           std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsFloorSupportedRef(const TensorInfo& input,
                         const TensorInfo& output,
                         std::string* reasonIfUnsupported)
{
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &FalseFuncU8<>);
}

}
