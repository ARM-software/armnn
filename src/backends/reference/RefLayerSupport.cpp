//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefLayerSupport.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/Types.hpp>
#include <armnn/Descriptors.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <LayerSupportCommon.hpp>
#include <backendsCommon/LayerSupportRules.hpp>

#include <vector>
#include <array>

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
                                         &FalseFunc<Params...>,
                                         &FalseFunc<Params...>,
                                         std::forward<Params>(params)...);
}

} // anonymous namespace

namespace
{

std::string CreateIncorrectDimensionsErrorMsg(unsigned int expected,
                                              unsigned int actual,
                                              std::string& layerStr,
                                              std::string& tensorName)
{
    std::string errorMsg = "Reference " + layerStr + ": Expected " + std::to_string(expected) + " dimensions but got" +
                           " " + std::to_string(actual) + " dimensions instead, for the '" + tensorName + "' tensor.";

    return errorMsg;
}

} // anonymous namespace

bool RefLayerSupport::IsAbsSupported(const TensorInfo& input, const TensorInfo& output,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    return IsElementwiseUnarySupported(input,
                                       output,
                                       ElementwiseUnaryDescriptor(UnaryOperation::Abs),
                                       reasonIfUnsupported);
}

bool RefLayerSupport::IsActivationSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const ActivationDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
   bool supported = true;

    // Define supported types.
    std::array<DataType,6> supportedTypes = {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference activation: input type not supported.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference activation: output type not supported.");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference activation: input and output types mismatched.");

    supported &= CheckSupportRule(ShapesAreSameRank(input, output), reasonIfUnsupported,
                                  "Reference activation: input and output shapes are of different rank.");


    struct ActivationFunctionSupported : public Rule
    {
        ActivationFunctionSupported(const ActivationDescriptor& desc)
        {
            switch(desc.m_Function)
            {
                case ActivationFunction::Abs:
                case ActivationFunction::BoundedReLu:
                case ActivationFunction::Elu:
                case ActivationFunction::HardSwish:
                case ActivationFunction::LeakyReLu:
                case ActivationFunction::Linear:
                case ActivationFunction::ReLu:
                case ActivationFunction::Sigmoid:
                case ActivationFunction::SoftReLu:
                case ActivationFunction::Sqrt:
                case ActivationFunction::Square:
                case ActivationFunction::TanH:
                {
                    m_Res = true;
                    break;
                }
                default:
                {
                    m_Res = false;
                    break;
                }
            }
        }
    };

    // Function is supported
    supported &= CheckSupportRule(ActivationFunctionSupported(descriptor), reasonIfUnsupported,
                                  "Reference activation: function not supported.");

    return supported;
}

bool RefLayerSupport::IsAdditionSupported(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    bool supported = true;

    std::array<DataType,7> supportedTypes = {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    supported &= CheckSupportRule(TypeAnyOf(input0, supportedTypes), reasonIfUnsupported,
                                  "Reference addition: input 0 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(input1, supportedTypes), reasonIfUnsupported,
                                  "Reference addition: input 1 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference addition: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input0, input1), reasonIfUnsupported,
                                  "Reference addition: input 0 and Input 1 types are mismatched");

    supported &= CheckSupportRule(TypesAreEqual(input0, output), reasonIfUnsupported,
                                  "Reference addition: input and output types are mismatched");

    supported &= CheckSupportRule(ShapesAreBroadcastCompatible(input0, input1, output), reasonIfUnsupported,
                                  "Reference addition: shapes are not suitable for implicit broadcast.");

    return supported;
}

bool RefLayerSupport::IsArgMinMaxSupported(const armnn::TensorInfo &input, const armnn::TensorInfo &output,
                                           const armnn::ArgMinMaxDescriptor &descriptor,
                                           armnn::Optional<std::string &> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);

    std::array<DataType, 7> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    bool supported = true;

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference ArgMinMax: input is not a supported type.");
    supported &= CheckSupportRule(TypeIs(output, DataType::Signed32), reasonIfUnsupported,
                                  "Reference ArgMinMax: output type not supported");

    return supported;
}

bool RefLayerSupport::IsBatchNormalizationSupported(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    const TensorInfo& mean,
                                                    const TensorInfo& variance,
                                                    const TensorInfo& beta,
                                                    const TensorInfo& gamma,
                                                    const BatchNormalizationDescriptor& descriptor,
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);

    std::array<DataType, 6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    bool supported = true;

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference batch normalization: input is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference batch normalization: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference batch normalization: input and output types are mismatched");

    supported &= CheckSupportRule(TypeAnyOf(mean, supportedTypes), reasonIfUnsupported,
                                  "Reference batch normalization: mean is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(variance, supportedTypes), reasonIfUnsupported,
                                  "Reference batch normalization: variance is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(beta, supportedTypes), reasonIfUnsupported,
                                  "Reference batch normalization: beta is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(gamma, supportedTypes), reasonIfUnsupported,
                                  "Reference batch normalization: gamma is not a supported type.");

    return supported;
}

bool RefLayerSupport::IsBatchToSpaceNdSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const BatchToSpaceNdDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);

    bool supported = true;

    std::string batchToSpaceNdLayerStr = "batchToSpaceNd";
    std::string inputTensorStr = "input";
    std::string outputTensorStr = "output";

    // Define supported types.
    std::array<DataType,6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference BatchToSpaceNd: input type not supported.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference BatchToSpaceNd: output type not supported.");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference BatchToSpaceNd: input and output types mismatched.");

    supported &= CheckSupportRule(TensorNumDimensionsAreCorrect(output, 4),
                                  reasonIfUnsupported,
                                  CreateIncorrectDimensionsErrorMsg(4,
                                                                    output.GetNumDimensions(),
                                                                    batchToSpaceNdLayerStr,
                                                                    outputTensorStr).data());

    supported &= CheckSupportRule(TensorNumDimensionsAreCorrect(input, 4),
                                  reasonIfUnsupported,
                                  CreateIncorrectDimensionsErrorMsg(4,
                                                                    input.GetNumDimensions(),
                                                                    batchToSpaceNdLayerStr,
                                                                    inputTensorStr).data());

    return supported;
}

bool RefLayerSupport::IsComparisonSupported(const TensorInfo& input0,
                                            const TensorInfo& input1,
                                            const TensorInfo& output,
                                            const ComparisonDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    std::array<DataType, 8> supportedInputTypes =
    {
        DataType::Boolean,
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    bool supported = true;
    supported &= CheckSupportRule(TypeAnyOf(input0, supportedInputTypes), reasonIfUnsupported,
                                  "Reference comparison: input 0 is not a supported type");

    supported &= CheckSupportRule(TypesAreEqual(input0, input1), reasonIfUnsupported,
                                  "Reference comparison: input 0 and Input 1 types are mismatched");

    supported &= CheckSupportRule(TypeIs(output, DataType::Boolean), reasonIfUnsupported,
                                  "Reference comparison: output is not of type Boolean");

    return supported;
}

bool RefLayerSupport::IsConcatSupported(const std::vector<const TensorInfo*> inputs,
                                        const TensorInfo& output,
                                        const ConcatDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);

    bool supported = true;
    std::array<DataType,6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference concatenation: output type not supported");
    for (const TensorInfo* input : inputs)
    {
        ARMNN_ASSERT(input != nullptr);
        supported &= CheckSupportRule(TypeAnyOf(*input, supportedTypes), reasonIfUnsupported,
            "Reference concatenation: input type not supported");

        supported &= CheckSupportRule(TypesAreEqual(*input, output), reasonIfUnsupported,
            "Reference concatenation: input and output types mismatched.");
    }

    return supported;
}

bool RefLayerSupport::IsConstantSupported(const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    std::array<DataType,8> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    return CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference constant: output is not a supported type.");
}

bool RefLayerSupport::IsConvertBf16ToFp32Supported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    bool supported = true;

    supported &= CheckSupportRule(TypeIs(input, DataType::BFloat16), reasonIfUnsupported,
                                  "Reference for ConvertBf16ToFp32 layer: input type not supported");

    supported &= CheckSupportRule(TypeIs(output, DataType::Float32), reasonIfUnsupported,
                                  "Reference for ConvertBf16ToFp32 layer: output type not supported");

    return supported;
}

bool RefLayerSupport::IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    return (IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                          input.GetDataType(),
                                          &TrueFunc<>,
                                          &FalseInputFuncF32<>,
                                          &FalseFuncU8<>,
                                          &FalseFuncI32<>,
                                          &FalseFuncU8<>) &&
            IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                          output.GetDataType(),
                                          &FalseOutputFuncF16<>,
                                          &TrueFunc<>,
                                          &FalseFuncU8<>,
                                          &FalseFuncI32<>,
                                          &FalseFuncU8<>));
}

bool RefLayerSupport::IsConvertFp32ToBf16Supported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    bool supported = true;

    supported &= CheckSupportRule(TypeIs(input, DataType::Float32), reasonIfUnsupported,
                                  "Reference for ConvertFp32ToBf16 layer: input type not supported");

    supported &= CheckSupportRule(TypeIs(output, DataType::BFloat16), reasonIfUnsupported,
                                  "Reference for ConvertFp32ToBf16 layer: output type not supported");

    return supported;
}

bool RefLayerSupport::IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    return (IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                          input.GetDataType(),
                                          &FalseInputFuncF16<>,
                                          &TrueFunc<>,
                                          &FalseFuncU8<>,
                                          &FalseFuncI32<>,
                                          &FalseFuncU8<>) &&
            IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                          output.GetDataType(),
                                          &TrueFunc<>,
                                          &FalseOutputFuncF32<>,
                                          &FalseFuncU8<>,
                                          &FalseFuncI32<>,
                                          &FalseFuncU8<>));
}

bool RefLayerSupport::IsConvolution2dSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const Convolution2dDescriptor& descriptor,
                                               const TensorInfo& weights,
                                               const Optional<TensorInfo>& biases,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    bool supported = true;

    // Define supported types.
    std::array<DataType,7> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference Convolution2d: input is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference Convolution2d: output is not a supported type.");

    // For Convolution2d, we allow to have BFloat16 input with Float32 output for optimization.
    if (input.GetDataType() == DataType::BFloat16)
    {
        if (output.GetDataType() != DataType::BFloat16 && output.GetDataType() != DataType::Float32)
        {
            reasonIfUnsupported.value() += "Output tensor type must be BFloat16 or Float32 for BFloat16 input.\n";
            supported = false;
        }
    }
    else
    {
        supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference Convolution2d: input and output types mismatched.");
    }

    const DataType inputType = input.GetDataType();
    if (IsQuantized8BitType(inputType))
    {
        ARMNN_NO_DEPRECATE_WARN_BEGIN
        std::array<DataType, 4> supportedWeightTypes =
        {
            DataType::QAsymmS8,
            DataType::QAsymmU8,
            DataType::QSymmS8,
            DataType::QuantizedSymm8PerAxis // deprecated
        };
        ARMNN_NO_DEPRECATE_WARN_END

        supported &= CheckSupportRule(TypeAnyOf(weights, supportedWeightTypes), reasonIfUnsupported,
                                      "Reference Convolution2d: weights type not supported for quantized input.");
    }
    else
    {
        supported &= CheckSupportRule(TypeAnyOf(weights, supportedTypes), reasonIfUnsupported,
                                      "Reference Convolution2d: weights is not a supported type.");

        supported &= CheckSupportRule(TypesAreEqual(input, weights), reasonIfUnsupported,
                                      "Reference Convolution2d: input and weights types mismatched.");
    }

    if (biases.has_value())
    {
        std::array<DataType,4> biasesSupportedTypes =
        {
            DataType::BFloat16,
            DataType::Float32,
            DataType::Float16,
            DataType::Signed32
        };

        supported &= CheckSupportRule(TypeAnyOf(biases.value(), biasesSupportedTypes), reasonIfUnsupported,
                                      "Reference Convolution2d: biases is not a supported type.");
    }
    IgnoreUnused(descriptor);

    return supported;
}

bool RefLayerSupport::IsDebugSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    bool supported = true;

    std::array<DataType, 8> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference for Debug layer: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference for Debug layer: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference for Debug layer: input and output types are mismatched");

    return supported;
}

bool RefLayerSupport::IsDepthToSpaceSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const DepthToSpaceDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    bool supported = true;

    std::array<DataType,6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
        "Reference DepthToSpace: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
        "Reference DepthToSpace: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
        "Reference DepthToSpace: input and output types are mismatched");

    return supported;
}

bool RefLayerSupport::IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const DepthwiseConvolution2dDescriptor& descriptor,
                                                      const TensorInfo& weights,
                                                      const Optional<TensorInfo>& biases,
                                                      Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    bool supported = true;

    // Define supported types.
    std::array<DataType,7> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference DepthwiseConvolution2d: input is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference DepthwiseConvolution2d: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference DepthwiseConvolution2d: input and output types mismatched.");

    const DataType inputType = input.GetDataType();
    if (IsQuantized8BitType(inputType))
    {
        ARMNN_NO_DEPRECATE_WARN_BEGIN
        std::array<DataType, 4> supportedWeightTypes =
                {
                        DataType::QAsymmS8,
                        DataType::QAsymmU8,
                        DataType::QSymmS8,
                        DataType::QuantizedSymm8PerAxis // deprecated
                };
        ARMNN_NO_DEPRECATE_WARN_END

        supported &= CheckSupportRule(TypeAnyOf(weights, supportedWeightTypes), reasonIfUnsupported,
                                       "Reference DepthwiseConvolution2d: weights type not supported for "
                                       "quantized input.");
    }
    else
    {
        supported &= CheckSupportRule(TypeAnyOf(weights, supportedTypes), reasonIfUnsupported,
                                      "Reference DepthwiseConvolution2d: weights is not a supported type.");

        supported &= CheckSupportRule(TypesAreEqual(input, weights), reasonIfUnsupported,
                                      "Reference DepthwiseConvolution2d: input and weights types mismatched.");
    }

    if (biases.has_value())
    {
        std::array<DataType,4> biasesSupportedTypes =
        {
            DataType::BFloat16,
            DataType::Float32,
            DataType::Float16,
            DataType::Signed32
        };
        supported &= CheckSupportRule(TypeAnyOf(biases.value(), biasesSupportedTypes), reasonIfUnsupported,
                                      "Reference DepthwiseConvolution2d: biases is not a supported type.");
    }

    return supported;

}

bool RefLayerSupport::IsDequantizeSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            Optional<std::string&> reasonIfUnsupported) const
{
   bool supported = true;

    std::array<DataType,4> supportedInputTypes = {
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedInputTypes), reasonIfUnsupported,
                                  "Reference for Dequantize layer: input type not supported.");

    supported &= CheckSupportRule( TypeNotPerAxisQuantized(input), reasonIfUnsupported,
                                    "Reference for Dequantize layer: per-axis quantized input not support .");

    supported &= CheckSupportRule(TypeNotPerAxisQuantized(input), reasonIfUnsupported,
                                  "Reference dequantize: per-axis quantized input not support .");

    std::array<DataType,3> supportedOutputTypes = {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16
    };

    supported &= CheckSupportRule(TypeAnyOf(output, supportedOutputTypes), reasonIfUnsupported,
                                  "Reference for Dequantize layer: output type not supported.");

    supported &= CheckSupportRule(ShapesAreSameTotalSize(input, output), reasonIfUnsupported,
                                  "Reference for Dequantize layer: input/output shapes have different num total "
                                  "elements.");

    return supported;
}

bool RefLayerSupport::IsDetectionPostProcessSupported(const TensorInfo& boxEncodings,
                                                      const TensorInfo& scores,
                                                      const TensorInfo& anchors,
                                                      const TensorInfo& detectionBoxes,
                                                      const TensorInfo& detectionClasses,
                                                      const TensorInfo& detectionScores,
                                                      const TensorInfo& numDetections,
                                                      const DetectionPostProcessDescriptor& descriptor,
                                                      Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(anchors, detectionBoxes, detectionClasses, detectionScores, numDetections, descriptor);

    bool supported = true;

    std::array<DataType,6> supportedInputTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(boxEncodings, supportedInputTypes), reasonIfUnsupported,
                                  "Reference DetectionPostProcess: input 0 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(scores, supportedInputTypes), reasonIfUnsupported,
                                  "Reference DetectionPostProcess: input 1 is not a supported type.");

    return supported;
}

bool RefLayerSupport::IsDilatedDepthwiseConvolutionSupported(const TensorInfo& input,
                                                             const TensorInfo& output,
                                                             const DepthwiseConvolution2dDescriptor& descriptor,
                                                             const TensorInfo& weights,
                                                             const Optional<TensorInfo>& biases,
                                                             Optional<std::string&> reasonIfUnsupported) const
{
    return IsDepthwiseConvolutionSupported(input, output, descriptor, weights, biases, reasonIfUnsupported);
}

bool RefLayerSupport::IsDivisionSupported(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    bool supported = true;

    std::array<DataType,7> supportedTypes = {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    supported &= CheckSupportRule(TypeAnyOf(input0, supportedTypes), reasonIfUnsupported,
                                  "Reference division: input 0 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(input1, supportedTypes), reasonIfUnsupported,
                                  "Reference division: input 1 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference division: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input0, input1), reasonIfUnsupported,
                                  "Reference division: input 0 and Input 1 types are mismatched");

    supported &= CheckSupportRule(TypesAreEqual(input0, output), reasonIfUnsupported,
                                  "Reference division: input and output types are mismatched");

    supported &= CheckSupportRule(ShapesAreBroadcastCompatible(input0, input1, output), reasonIfUnsupported,
                                  "Reference division: shapes are not suitable for implicit broadcast.");

    return supported;
}

bool RefLayerSupport::IsElementwiseUnarySupported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const ElementwiseUnaryDescriptor& descriptor,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);

    std::array<DataType, 7> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    std::array<DataType, 1> logicalSupportedTypes =
    {
        DataType::Boolean
    };

    bool supported = true;

    if (descriptor.m_Operation == UnaryOperation::LogicalNot)
    {
        supported &= CheckSupportRule(TypeAnyOf(input, logicalSupportedTypes), reasonIfUnsupported,
                                      "Reference elementwise unary: input type not supported");

        supported &= CheckSupportRule(TypeAnyOf(output, logicalSupportedTypes), reasonIfUnsupported,
                                      "Reference elementwise unary: output type not supported");
    }
    else
    {
        supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                      "Reference elementwise unary: input type not supported");

        supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                      "Reference elementwise unary: output type not supported");
    }

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference elementwise unary: input and output types not matching");

    supported &= CheckSupportRule(ShapesAreSameTotalSize(input, output), reasonIfUnsupported,
                                  "Reference elementwise unary: input and output shapes"
                                  "have different number of total elements");

    return supported;
}

bool RefLayerSupport::IsEqualSupported(const TensorInfo& input0,
                                       const TensorInfo& input1,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return IsComparisonSupported(input0,
                                 input1,
                                 output,
                                 ComparisonDescriptor(ComparisonOperation::Equal),
                                 reasonIfUnsupported);
}

bool RefLayerSupport::IsFakeQuantizationSupported(const TensorInfo& input,
                                                  const FakeQuantizationDescriptor& descriptor,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    bool supported = true;

    std::array<DataType,1> supportedTypes =
    {
        DataType::Float32
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference fake quantization: input type not supported.");

    return supported;
}

bool RefLayerSupport::IsFillSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const FillDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    IgnoreUnused(output);

    bool supported = true;

    std::array<DataType,3> supportedTypes =
    {
        DataType::Float32,
        DataType::Float16,
        DataType::Signed32
    };

    supported &= CheckSupportRule(TypeIs(input, DataType::Signed32), reasonIfUnsupported,
                                  "Reference Fill: input type not supported.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference Fill: output type not supported.");
    return supported;
}

bool RefLayerSupport::IsFloorSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(output);
    bool supported = true;

    std::array<DataType,3> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference Floor: input type not supported.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference Floor: output type not supported.");

    return supported;
}

bool RefLayerSupport::IsFullyConnectedSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const TensorInfo& weights,
                                                const TensorInfo& biases,
                                                const FullyConnectedDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    bool supported = true;

    // Define supported types.
    std::array<DataType,6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference Fully Connected: input type not supported.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference Fully Connected: output type not supported.");

    supported &= CheckSupportRule(TypeAnyOf(weights, supportedTypes), reasonIfUnsupported,
                                  "Reference Fully Connected: weights type not supported.");

    // For FullyConnected, we allow to have BFloat16 input with Float32 output for optimization.
    if (input.GetDataType() == DataType::BFloat16)
    {
        if (output.GetDataType() != DataType::BFloat16 && output.GetDataType() != DataType::Float32)
        {
            reasonIfUnsupported.value() += "Output tensor type must be BFloat16 or Float32 for BFloat16 input.\n";
            supported = false;
        }
    }
    else
    {
        supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference Fully Connected: input and output types mismatched.");
    }

    supported &= CheckSupportRule(TypeAnyOf(weights, supportedTypes), reasonIfUnsupported,
                                  "Reference Fully Connected: weights is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input, weights), reasonIfUnsupported,
                                  "Reference Fully Connected: input and weights types mismatched.");

    if (descriptor.m_BiasEnabled)
    {
        // Defined supported types for bias
        std::array<DataType, 5>
        supportedBiasTypes =
        {
            DataType::BFloat16,
            DataType::Float32,
            DataType::Float16,
            DataType::Signed32,
            DataType::QAsymmS8
        };

        supported &= CheckSupportRule(TypeAnyOf(biases, supportedBiasTypes), reasonIfUnsupported,
                                      "Reference Fully Connected: bias type not supported.");

        supported &= CheckSupportRule(BiasAndWeightsTypesMatch(biases, weights), reasonIfUnsupported,
                                      "Reference Fully Connected: bias and weight types mismatch.");

        supported &= CheckSupportRule(BiasAndWeightsTypesCompatible(weights, supportedBiasTypes), reasonIfUnsupported,
                                      "Reference Fully Connected: bias type inferred from weights is incompatible.");

        supported &= CheckSupportRule(TensorNumDimensionsAreCorrect(biases, 1U), reasonIfUnsupported,
                                      "Reference Fully Connected: bias must have 1 dimension.");

    }

    return supported;
}

bool RefLayerSupport::IsGatherSupported(const armnn::TensorInfo& input0,
                                        const armnn::TensorInfo& input1,
                                        const armnn::TensorInfo& output,
                                        const GatherDescriptor& descriptor,
                                        armnn::Optional<std::string&> reasonIfUnsupported) const
{
    bool supported = true;
    std::array<DataType,7> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    if (descriptor.m_Axis != 0)
    {
        reasonIfUnsupported.value() += std::string("Reference Gather: axis not supported\n");
        supported &= false;
    }
    supported &= CheckSupportRule(TypeAnyOf(input0, supportedTypes), reasonIfUnsupported,
                                  "Reference Gather: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference Gather: output type not supported");

    supported &= CheckSupportRule(TypeIs(input1, DataType::Signed32), reasonIfUnsupported,
                                  "Reference Gather: indices (input1) type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input0, output), reasonIfUnsupported,
                                  "Reference Gather: input and output types not matching");

    return supported;
}

bool RefLayerSupport::IsGreaterSupported(const TensorInfo& input0,
                                         const TensorInfo& input1,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return IsComparisonSupported(input0,
                                 input1,
                                 output,
                                 ComparisonDescriptor(ComparisonOperation::Greater),
                                 reasonIfUnsupported);
}

bool RefLayerSupport::IsInputSupported(const TensorInfo& /*input*/,
                                       Optional<std::string&> /*reasonIfUnsupported*/) const
{
    return true;
}

bool RefLayerSupport::IsInstanceNormalizationSupported(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const InstanceNormalizationDescriptor& descriptor,
                                                       Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    // Define supported types
    std::array<DataType, 3> supportedTypes =
        {
            DataType::BFloat16,
            DataType::Float32,
            DataType::Float16
        };

    bool supported = true;

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference Instance Normalization: input type not supported.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference Instance Normalization: output type not supported.");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference Instance Normalization: input and output types mismatched.");

    supported &= CheckSupportRule(ShapesAreSameTotalSize(input, output), reasonIfUnsupported,
                                  "Reference Instance Normalization: input and output shapes have different "
                                  "num total elements.");

    return supported;
}

bool RefLayerSupport::IsL2NormalizationSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const L2NormalizationDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    // Define supported types
    std::array<DataType, 6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    bool supported = true;

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference L2normalization: input type not supported.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference L2normalization: output type not supported.");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference L2normalization: input and output types mismatched.");

    supported &= CheckSupportRule(ShapesAreSameTotalSize(input, output), reasonIfUnsupported,
                                  "Reference L2normalization: input and output shapes have different "
                                  "num total elements.");

    return supported;
}

bool RefLayerSupport::IsLogicalBinarySupported(const TensorInfo& input0,
                                               const TensorInfo& input1,
                                               const TensorInfo& output,
                                               const LogicalBinaryDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);

    std::array<DataType, 1> supportedTypes =
    {
        DataType::Boolean
    };

    bool supported = true;
    supported &= CheckSupportRule(TypeAnyOf(input0, supportedTypes), reasonIfUnsupported,
                                  "Reference LogicalBinary: input 0 type not supported");
    supported &= CheckSupportRule(TypeAnyOf(input1, supportedTypes), reasonIfUnsupported,
                                  "Reference LogicalBinary: input 1 type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input0, output), reasonIfUnsupported,
                                  "Reference LogicalBinary: input and output types do not match");

    return supported;
}

bool RefLayerSupport::IsLogSoftmaxSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const LogSoftmaxDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);

    std::array<DataType, 3> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16
    };

    bool supported = true;
    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference LogSoftmax: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference LogSoftmax: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference LogSoftmax: input and output types do not match");

    return supported;
}

bool RefLayerSupport::IsLstmSupported(const TensorInfo& input,
                                      const TensorInfo& outputStateIn,
                                      const TensorInfo& cellStateIn,
                                      const TensorInfo& scratchBuffer,
                                      const TensorInfo& outputStateOut,
                                      const TensorInfo& cellStateOut,
                                      const TensorInfo& output,
                                      const LstmDescriptor& descriptor,
                                      const LstmInputParamsInfo& paramsInfo,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    IgnoreUnused(paramsInfo);

    bool supported = true;

    std::array<DataType,3> supportedTypes = {
        DataType::BFloat16,
        DataType::Float32,
        DataType::QSymmS16
    };

    // check inputs and outputs
    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference Lstm: input is not a supported type.");
    supported &= CheckSupportRule(TypesAreEqual(input, outputStateIn), reasonIfUnsupported,
                                  "Reference Lstm: input and outputStateIn types are mismatched");
    supported &= CheckSupportRule(TypesAreEqual(input, cellStateIn), reasonIfUnsupported,
                                  "Reference Lstm: input and cellStateIn types are mismatched");
    supported &= CheckSupportRule(TypesAreEqual(input, scratchBuffer), reasonIfUnsupported,
                                  "Reference Lstm: input and scratchBuffer types are mismatched");
    supported &= CheckSupportRule(TypesAreEqual(input, outputStateOut), reasonIfUnsupported,
                                  "Reference Lstm: input and outputStateOut types are mismatched");
    supported &= CheckSupportRule(TypesAreEqual(input, cellStateOut), reasonIfUnsupported,
                                  "Reference Lstm: input and cellStateOut types are mismatched");
    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference Lstm: input and output types are mismatched");
    // check layer parameters
    supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetInputToForgetWeights()), reasonIfUnsupported,
                                  "Reference Lstm: input and InputToForgetWeights types are mismatched");
    supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetInputToCellWeights()), reasonIfUnsupported,
                                  "Reference Lstm: input and InputToCellWeights types are mismatched");
    supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetInputToOutputWeights()), reasonIfUnsupported,
                                  "Reference Lstm: input and InputToOutputWeights types are mismatched");
    supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetRecurrentToForgetWeights()), reasonIfUnsupported,
                                  "Reference Lstm: input and RecurrentToForgetWeights types are mismatched");
    supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetRecurrentToCellWeights()), reasonIfUnsupported,
                                  "Reference Lstm: input and RecurrentToCellWeights types are mismatched");
    supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetRecurrentToOutputWeights()), reasonIfUnsupported,
                                  "Reference Lstm: input and RecurrentToOutputWeights types are mismatched");
    supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetForgetGateBias()), reasonIfUnsupported,
                                  "Reference Lstm: input and ForgetGateBias types are mismatched");
    supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetCellBias()), reasonIfUnsupported,
                                  "Reference Lstm: input and CellBias types are mismatched");
    supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetOutputGateBias()), reasonIfUnsupported,
                                  "Reference Lstm: input and OutputGateBias types are mismatched");
    if (!descriptor.m_CifgEnabled)
    {
        supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetInputToInputWeights()), reasonIfUnsupported,
                                      "Reference Lstm: input and InputToInputWeights types are mismatched");
        supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetRecurrentToInputWeights()),
                                      reasonIfUnsupported,
                                      "Reference Lstm: input and RecurrentToInputWeights types are mismatched");
        supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetInputGateBias()), reasonIfUnsupported,
                                      "Reference Lstm: input and InputGateBias types are mismatched");
        if (descriptor.m_PeepholeEnabled)
        {
            supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetCellToInputWeights()),
                                          reasonIfUnsupported,
                                          "Reference Lstm: input and CellToInputWeights types are mismatched");
        }
    }
    if (descriptor.m_PeepholeEnabled)
    {
        supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetCellToForgetWeights()), reasonIfUnsupported,
                                      "Reference Lstm: input and CellToForgetWeights types are mismatched");
        supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetCellToOutputWeights()), reasonIfUnsupported,
                                      "Reference Lstm: input and CellToOutputWeights types are mismatched");
    }
    if (descriptor.m_ProjectionEnabled)
    {
        supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetProjectionWeights()), reasonIfUnsupported,
                                      "Reference Lstm: input and mProjectionWeights types are mismatched");
        if (paramsInfo.m_ProjectionBias != nullptr)
        {
            supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetProjectionBias()), reasonIfUnsupported,
                                          "Reference Lstm: input and ProjectionBias types are mismatched");
        }
    }
    if (descriptor.m_LayerNormEnabled)
    {
        if (!descriptor.m_CifgEnabled)
        {
            supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetInputLayerNormWeights()),
                                          reasonIfUnsupported,
                                          "Reference Lstm: input and InputLayerNormWeights types are mismatched");
        }
        supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetForgetLayerNormWeights()),
                                      reasonIfUnsupported,
                                      "Reference Lstm: input and ForgetLayerNormWeights types are mismatched");
        supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetCellLayerNormWeights()),
                                      reasonIfUnsupported,
                                      "Reference Lstm: input and CellLayerNormWeights types are mismatched");
        supported &= CheckSupportRule(TypesAreEqual(input, paramsInfo.GetOutputLayerNormWeights()),
                                      reasonIfUnsupported,
                                      "Reference Lstm: input and OutputLayerNormWeights types are mismatched");
    }

    return supported;
}

bool RefLayerSupport::IsMaximumSupported(const TensorInfo& input0,
                                         const TensorInfo& input1,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    bool supported = true;

    std::array<DataType,7> supportedTypes = {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    supported &= CheckSupportRule(TypeAnyOf(input0, supportedTypes), reasonIfUnsupported,
                                  "Reference maximum: input 0 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(input1, supportedTypes), reasonIfUnsupported,
                                  "Reference maximum: input 1 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference maximum: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input0, input1), reasonIfUnsupported,
                                  "Reference maximum: input 0 and Input 1 types are mismatched");

    supported &= CheckSupportRule(TypesAreEqual(input0, output), reasonIfUnsupported,
                                  "Reference maximum: input and output types are mismatched");

    supported &= CheckSupportRule(ShapesAreBroadcastCompatible(input0, input1, output), reasonIfUnsupported,
                                  "Reference maximum: shapes are not suitable for implicit broadcast.");

    return supported;
}

bool RefLayerSupport::IsMeanSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const MeanDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    bool supported = true;
    std::string meanLayerStr = "Mean";
    std::string outputTensorStr = "output";

    std::array<DataType,6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference Mean: input type not supported.");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference Mean: input and output types are mismatched");

    if (descriptor.m_KeepDims)
    {
        supported &= CheckSupportRule(TensorNumDimensionsAreCorrect(output, input.GetNumDimensions()),
                                      reasonIfUnsupported,
                                      CreateIncorrectDimensionsErrorMsg(input.GetNumDimensions(),
                                                                        output.GetNumDimensions(),
                                                                        meanLayerStr, outputTensorStr).data());
    }
    else if (descriptor.m_Axis.empty())
    {
        supported &= CheckSupportRule(TensorNumDimensionsAreCorrect(output, 1),
                                      reasonIfUnsupported,
                                      CreateIncorrectDimensionsErrorMsg(1, output.GetNumDimensions(),
                                                                        meanLayerStr, outputTensorStr).data());
    }
    else
    {
        auto outputDim = input.GetNumDimensions() - armnn::numeric_cast<unsigned int>(descriptor.m_Axis.size());

        if (outputDim > 0)
        {
            supported &= CheckSupportRule(TensorNumDimensionsAreCorrect(output, outputDim),
                                          reasonIfUnsupported,
                                          CreateIncorrectDimensionsErrorMsg(outputDim, output.GetNumDimensions(),
                                                                            meanLayerStr, outputTensorStr).data());
        }
        else
        {
            supported &= CheckSupportRule(TensorNumDimensionsAreCorrect(output, 1),
                                          reasonIfUnsupported,
                                          CreateIncorrectDimensionsErrorMsg(1, output.GetNumDimensions(),
                                                                            meanLayerStr, outputTensorStr).data());
        }
    }

    return supported;
}

bool RefLayerSupport::IsMergerSupported(const std::vector<const TensorInfo*> inputs,
                                        const TensorInfo& output,
                                        const MergerDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return IsConcatSupported(inputs, output, descriptor, reasonIfUnsupported);
}

bool RefLayerSupport::IsMemCopySupported(const TensorInfo &input,
                                         const TensorInfo &output,
                                         Optional<std::string &> reasonIfUnsupported) const
{
    bool supported = true;

    std::array<DataType,7> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Boolean
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference MemCopy: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference MemCopy: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference MemCopy: input and output types are mismatched");

    return supported;
}

bool RefLayerSupport::IsMinimumSupported(const TensorInfo& input0,
                                         const TensorInfo& input1,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    bool supported = true;

    std::array<DataType,7> supportedTypes = {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    supported &= CheckSupportRule(TypeAnyOf(input0, supportedTypes), reasonIfUnsupported,
                                  "Reference minimum: input 0 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(input1, supportedTypes), reasonIfUnsupported,
                                  "Reference minimum: input 1 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference minimum: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input0, input1), reasonIfUnsupported,
                                  "Reference minimum: input 0 and Input 1 types are mismatched");

    supported &= CheckSupportRule(TypesAreEqual(input0, output), reasonIfUnsupported,
                                  "Reference minimum: input and output types are mismatched");

    supported &= CheckSupportRule(ShapesAreBroadcastCompatible(input0, input1, output), reasonIfUnsupported,
                                  "Reference minimum: shapes are not suitable for implicit broadcast.");

    return supported;
}

bool RefLayerSupport::IsMultiplicationSupported(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    bool supported = true;

    std::array<DataType,7> supportedTypes = {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    supported &= CheckSupportRule(TypeAnyOf(input0, supportedTypes), reasonIfUnsupported,
                                  "Reference multiplication: input 0 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(input1, supportedTypes), reasonIfUnsupported,
                                  "Reference multiplication: input 1 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference multiplication: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input0, input1), reasonIfUnsupported,
                                  "Reference multiplication: input 0 and Input 1 types are mismatched");

    supported &= CheckSupportRule(TypesAreEqual(input0, output), reasonIfUnsupported,
                                  "Reference multiplication: input and output types are mismatched");

    supported &= CheckSupportRule(ShapesAreBroadcastCompatible(input0, input1, output), reasonIfUnsupported,
                                  "Reference multiplication: shapes are not suitable for implicit broadcast.");

    return supported;
}

bool RefLayerSupport::IsNormalizationSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const NormalizationDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);

    // Define supported types
    std::array<DataType, 6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    bool supported = true;

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference normalization: input type not supported.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference normalization: output type not supported.");

    supported &= CheckSupportRule(ShapesAreSameTotalSize(input, output), reasonIfUnsupported,
                                  "Reference normalization: input and output shapes have different "
                                  "num total elements.");

    return supported;
}

bool RefLayerSupport::IsOutputSupported(const TensorInfo& /*output*/,
                                        Optional<std::string&> /*reasonIfUnsupported*/) const
{
    return true;
}

bool RefLayerSupport::IsPadSupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const PadDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    bool supported = true;

    // Define supported output and inputs types.
    std::array<DataType,6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference pad: input is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference pad: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference pad: input and output types are mismatched.");

    return supported;
}

bool RefLayerSupport::IsPermuteSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const PermuteDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    bool supported = true;

    // Define supported output and inputs types.
    std::array<DataType, 6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference permute: input is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference permute: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference permute: input and output types are mismatched.");

    return supported;
}

bool RefLayerSupport::IsPooling2dSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const Pooling2dDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    bool supported = true;

    // Define supported output and inputs types.
    std::array<DataType,6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference poolind2d: input is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference poolind2d: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference poolind2d: input and output types are mismatched.");

    return supported;
}

bool RefLayerSupport::IsQLstmSupported(const TensorInfo& input,
                                       const TensorInfo& previousOutputIn,
                                       const TensorInfo& previousCellStateIn,
                                       const TensorInfo& outputStateOut,
                                       const TensorInfo& cellStateOut,
                                       const TensorInfo& output,
                                       const QLstmDescriptor& descriptor,
                                       const LstmInputParamsInfo& paramsInfo,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input);
    IgnoreUnused(previousOutputIn);
    IgnoreUnused(previousCellStateIn);
    IgnoreUnused(outputStateOut);
    IgnoreUnused(cellStateOut);
    IgnoreUnused(output);
    IgnoreUnused(descriptor);
    IgnoreUnused(paramsInfo);

    IgnoreUnused(reasonIfUnsupported);

    return true;
}

bool RefLayerSupport::IsQuantizeSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
   bool supported = true;

    // Define supported input types.
    std::array<DataType,7> supportedInputTypes = {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedInputTypes), reasonIfUnsupported,
                                  "Reference quantize: input type not supported.");

    // Define supported output types.
    std::array<DataType,4> supportedOutputTypes = {
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS8,
        DataType::QSymmS16
    };
    supported &= CheckSupportRule(TypeAnyOf(output, supportedOutputTypes), reasonIfUnsupported,
                                  "Reference quantize: output type not supported.");

    supported &= CheckSupportRule(ShapesAreSameTotalSize(input, output), reasonIfUnsupported,
                                  "Reference quantize: input and output shapes have different num total elements.");

    return supported;
}

bool RefLayerSupport::IsRankSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input);
    // Define supported output types.
    std::array<DataType,1> supportedOutputTypes =
    {
        DataType::Signed32,
    };

    return CheckSupportRule(TypeAnyOf(output, supportedOutputTypes), reasonIfUnsupported,
           "Reference rank: input type not supported.");
}

bool RefLayerSupport::IsReshapeSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const ReshapeDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(output);
    IgnoreUnused(descriptor);
    // Define supported output types.
    std::array<DataType,8> supportedOutputTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::Signed32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Boolean
    };

    return CheckSupportRule(TypeAnyOf(input, supportedOutputTypes), reasonIfUnsupported,
        "Reference reshape: input type not supported.");
}

bool RefLayerSupport::IsResizeBilinearSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    bool supported = true;
    std::array<DataType,6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference ResizeBilinear: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference ResizeBilinear: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference ResizeBilinear: input and output types not matching");

    return supported;
}

bool RefLayerSupport::IsResizeSupported(const TensorInfo& input,
                                        const TensorInfo& output,
                                        const ResizeDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    bool supported = true;
    std::array<DataType,6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference Resize: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference Resize: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference Resize: input and output types not matching");

    return supported;
}

bool RefLayerSupport::IsRsqrtSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return IsElementwiseUnarySupported(input,
                                       output,
                                       ElementwiseUnaryDescriptor(UnaryOperation::Rsqrt),
                                       reasonIfUnsupported);
}

bool RefLayerSupport::IsSliceSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const SliceDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    bool supported = true;

    std::array<DataType, 5> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference Slice: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference Slice: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference Slice: input and output types are mismatched");

    return supported;
}

bool RefLayerSupport::IsSoftmaxSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const SoftmaxDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    bool supported = true;
    std::array<DataType,7> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QSymmS8,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference Softmax: output type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference Softmax: input type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference Softmax: input type not supported");

    return supported;
}

bool RefLayerSupport::IsSpaceToBatchNdSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const SpaceToBatchNdDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    bool supported = true;
    std::array<DataType,6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference SpaceToBatchNd: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference SpaceToBatchNd: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference SpaceToBatchNd: input and output types are mismatched");

    return supported;
}

bool RefLayerSupport::IsSpaceToDepthSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const SpaceToDepthDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{

    IgnoreUnused(descriptor);
    bool supported = true;

    std::array<DataType,6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
        "Reference SpaceToDepth: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
        "Reference SpaceToDepth: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
        "Reference SpaceToDepth: input and output types are mismatched");

    return supported;
}

bool RefLayerSupport::IsSplitterSupported(const TensorInfo& input,
                                          const ViewsDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    bool supported = true;
    std::array<DataType,6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference splitter: input type not supported");

    return supported;
}

bool RefLayerSupport::IsSplitterSupported(const TensorInfo& input,
                                          const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                                          const ViewsDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    bool supported = true;
    std::array<DataType,6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference splitter: output type not supported");
    for (const TensorInfo& output : outputs)
    {
        supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                      "Reference splitter: input type not supported");

        supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                      "Reference splitter: input and output types mismatched.");
    }

    return supported;
}

bool RefLayerSupport::IsStackSupported(const std::vector<const TensorInfo*>& inputs,
                                       const TensorInfo& output,
                                       const StackDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);

    bool supported = true;
    std::array<DataType,6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference stack: output type not supported");
    for (const TensorInfo* input : inputs)
    {
        ARMNN_ASSERT(input != nullptr);
        supported &= CheckSupportRule(TypeAnyOf(*input, supportedTypes), reasonIfUnsupported,
            "Reference stack: input type not supported");

        supported &= CheckSupportRule(TypesAreEqual(*input, output), reasonIfUnsupported,
            "Reference stack: input and output types mismatched.");
    }

    return supported;
}

bool RefLayerSupport::IsStridedSliceSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const StridedSliceDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    bool supported = true;

    std::array<DataType,5> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference StridedSlice: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference StridedSlice: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference StridedSlice: input and output types are mismatched");

    return supported;
}

bool RefLayerSupport::IsSubtractionSupported(const TensorInfo& input0,
                                             const TensorInfo& input1,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    bool supported = true;

    std::array<DataType,7> supportedTypes = {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16,
        DataType::Signed32
    };

    supported &= CheckSupportRule(TypeAnyOf(input0, supportedTypes), reasonIfUnsupported,
                                  "Reference subtraction: input 0 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(input1, supportedTypes), reasonIfUnsupported,
                                  "Reference subtraction: input 1 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference subtraction: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input0, input1), reasonIfUnsupported,
                                  "Reference subtraction: input 0 and Input 1 types are mismatched");

    supported &= CheckSupportRule(TypesAreEqual(input0, output), reasonIfUnsupported,
                                  "Reference subtraction: input and output types are mismatched");

    supported &= CheckSupportRule(ShapesAreBroadcastCompatible(input0, input1, output), reasonIfUnsupported,
                                  "Reference subtraction: shapes are not suitable for implicit broadcast.");

    return supported;
}

bool RefLayerSupport::IsPreluSupported(const TensorInfo& input,
                                       const TensorInfo& alpha,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    bool supported = true;

    std::array<DataType, 6> supportedTypes
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "PReLU: input is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(alpha, supportedTypes), reasonIfUnsupported,
                                  "PReLU: alpha is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "PReLU: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input, alpha, output), reasonIfUnsupported,
                                  "PReLU: input, alpha and output types are mismatched");

    supported &= CheckSupportRule(ShapesAreBroadcastCompatible(input, alpha, output), reasonIfUnsupported,
                                  "PReLU: shapes are not suitable for implicit broadcast");

    return supported;
}

bool RefLayerSupport::IsTransposeConvolution2dSupported(const TensorInfo& input,
                                                        const TensorInfo& output,
                                                        const TransposeConvolution2dDescriptor& descriptor,
                                                        const TensorInfo& weights,
                                                        const Optional<TensorInfo>& biases,
                                                        Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    bool supported = true;

    std::array<DataType,7> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference TransposeConvolution2d: input is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference TransposeConvolution2d: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference TransposeConvolution2d: input and output types mismatched.");


    const DataType inputType = input.GetDataType();
    if (IsQuantized8BitType(inputType))
    {
        ARMNN_NO_DEPRECATE_WARN_BEGIN
        std::array<DataType, 4> supportedWeightTypes =
        {
            DataType::QAsymmS8,
            DataType::QAsymmU8,
            DataType::QSymmS8,
            DataType::QuantizedSymm8PerAxis //Deprecated
        };
        ARMNN_NO_DEPRECATE_WARN_END

        supported &= CheckSupportRule(TypeAnyOf(weights, supportedWeightTypes), reasonIfUnsupported,
                                      "Reference TransposeConvolution2d: weights type not supported for "
                                      "quantized input.");
    }
    else
    {
        supported &= CheckSupportRule(TypeAnyOf(weights, supportedTypes), reasonIfUnsupported,
                                    "Reference TransposeConvolution2d: weights is not a supported type.");

        supported &= CheckSupportRule(TypesAreEqual(input, weights), reasonIfUnsupported,
                                    "Reference TransposeConvolution2d: input and weights types mismatched.");
    }

    if (biases.has_value())
    {
        std::array<DataType,4> biasesSupportedTypes =
        {
            DataType::BFloat16,
            DataType::Float32,
            DataType::Float16,
            DataType::Signed32
        };
        supported &= CheckSupportRule(TypeAnyOf(biases.value(), biasesSupportedTypes), reasonIfUnsupported,
                                      "Reference TransposeConvolution2d: biases is not a supported type.");
    }

    return supported;
}

bool RefLayerSupport::IsTransposeSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const TransposeDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    bool supported = true;

    // Define supported output and inputs types.
    std::array<DataType, 6> supportedTypes =
    {
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmS8,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference transpose: input is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference transpose: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference transpose: input and output types are mismatched.");

    return supported;
}

} // namespace armnn
