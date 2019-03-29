//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefLayerSupport.hpp"
#include "RefBackendId.hpp"

#include <InternalTypes.hpp>
#include <LayerSupportCommon.hpp>
#include <armnn/Types.hpp>
#include <armnn/Descriptors.hpp>

#include <backendsCommon/BackendRegistry.hpp>

#include <boost/core/ignore_unused.hpp>

#include <vector>
#include <algorithm>
#include <array>

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
                                         &FalseFunc<Params...>,
                                         &FalseFunc<Params...>,
                                         std::forward<Params>(params)...);
}

} // anonymous namespace


namespace
{
template<typename F>
bool CheckSupportRule(F rule, Optional<std::string&> reasonIfUnsupported, const char* reason)
{
    bool supported = rule();
    if (!supported && reason)
    {
        reasonIfUnsupported.value() += std::string(reason) + "\n"; // Append the reason on a new line
    }
    return supported;
}

struct Rule
{
    bool operator()() const
    {
        return m_Res;
    }

    bool m_Res = true;
};

template<typename T>
bool AllTypesAreEqualImpl(T t)
{
    return true;
}

template<typename T, typename... Rest>
bool AllTypesAreEqualImpl(T t1, T t2, Rest... rest)
{
    static_assert(std::is_same<T, TensorInfo>::value, "Type T must be a TensorInfo");

    return (t1.GetDataType() == t2.GetDataType()) && AllTypesAreEqualImpl(t2, rest...);
}

struct TypesAreEqual : public Rule
{
    template<typename ... Ts>
    TypesAreEqual(const Ts&... ts)
    {
        m_Res = AllTypesAreEqualImpl(ts...);
    }
};

struct QuantizationParametersAreEqual : public Rule
{
    QuantizationParametersAreEqual(const TensorInfo& info0, const TensorInfo& info1)
    {
        m_Res = info0.GetQuantizationScale() == info1.GetQuantizationScale() &&
                info0.GetQuantizationOffset() == info1.GetQuantizationOffset();
    }
};

struct TypeAnyOf : public Rule
{
    template<typename Container>
    TypeAnyOf(const TensorInfo& info, const Container& c)
    {
        m_Res = std::any_of(c.begin(), c.end(), [&info](DataType dt)
            {
                return dt == info.GetDataType();
            });
    }
};

struct ShapesAreSameRank : public Rule
{
    ShapesAreSameRank(const TensorInfo& info0, const TensorInfo& info1)
    {
        m_Res = info0.GetShape().GetNumDimensions() == info1.GetShape().GetNumDimensions();
    }
};

struct ShapesAreBroadcastCompatible : public Rule
{
    unsigned int CalcInputSize(const TensorShape& in, const TensorShape& out, unsigned int idx)
    {
        unsigned int offset = out.GetNumDimensions() - in.GetNumDimensions();
        unsigned int sizeIn = (idx < offset) ? 1 : in[idx-offset];
        return sizeIn;
    }

    ShapesAreBroadcastCompatible(const TensorInfo& in0, const TensorInfo& in1, const TensorInfo& out)
    {
        const TensorShape& shape0 = in0.GetShape();
        const TensorShape& shape1 = in1.GetShape();
        const TensorShape& outShape = out.GetShape();

        for (unsigned int i=0; i < outShape.GetNumDimensions() && m_Res; i++)
        {
            unsigned int sizeOut = outShape[i];
            unsigned int sizeIn0 = CalcInputSize(shape0, outShape, i);
            unsigned int sizeIn1 = CalcInputSize(shape1, outShape, i);

            m_Res &= ((sizeIn0 == sizeOut) || (sizeIn0 == 1)) &&
                     ((sizeIn1 == sizeOut) || (sizeIn1 == 1));
        }
    }
};
} // namespace


bool RefLayerSupport::IsActivationSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const ActivationDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
   bool supported = true;

    // Define supported types.
    std::array<DataType,2> supportedTypes = {
        DataType::Float32,
        DataType::QuantisedAsymm8
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

    std::array<DataType,2> supportedTypes = {
        DataType::Float32,
        DataType::QuantisedAsymm8
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
    return IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                         output.GetDataType(),
                                         &FalseFunc<>,
                                         &TrueFunc<>,
                                         &TrueFunc<>,
                                         &TrueFunc<>,
                                         &FalseFunc<>);
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
                                       Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
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

bool RefLayerSupport::IsDequantizeSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &FalseFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsDetectionPostProcessSupported(const armnn::TensorInfo& input0,
                                                      const armnn::TensorInfo& input1,
                                                      const armnn::DetectionPostProcessDescriptor& descriptor,
                                                      armnn::Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input1);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input0.GetDataType(),
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

bool RefLayerSupport::IsGatherSupported(const armnn::TensorInfo& input0,
                                        const armnn::TensorInfo& input1,
                                        const armnn::TensorInfo& output,
                                        armnn::Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsGreaterSupported(const TensorInfo& input0,
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

bool RefLayerSupport::IsMemCopySupported(const TensorInfo &input,
                                         const TensorInfo &output,
                                         Optional<std::string &> reasonIfUnsupported) const
{
    ignore_unused(output);
    return IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                         input.GetDataType(),
                                         &TrueFunc<>,
                                         &TrueFunc<>,
                                         &TrueFunc<>,
                                         &FalseFuncI32<>,
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
    return IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                         output.GetDataType(),
                                         &TrueFunc<>,
                                         &TrueFunc<>,
                                         &TrueFunc<>,
                                         &FalseFuncI32<>,
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
                                         const ReshapeDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsResizeBilinearSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool RefLayerSupport::IsRsqrtSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &FalseFuncU8<>);
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
