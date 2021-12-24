//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/LayerSupport.hpp>
#include <armnn/Optional.hpp>
#include <armnn/backends/ILayerSupport.hpp>
#include <armnn/BackendRegistry.hpp>

#include <armnn/backends/IBackendInternal.hpp>

#include <armnn/utility/Assert.hpp>

#include <cstring>
#include <algorithm>
#include <unordered_map>

namespace
{

/// Helper function to copy a full string to a truncated version.
void CopyErrorMessage(char* truncatedString, const char* fullString, size_t maxLength)
{
    if(truncatedString != nullptr)
    {
        std::snprintf(truncatedString, maxLength, "%s", fullString);
    }
}

} // anonymous namespace

namespace armnn
{

// Helper macro to avoid code duplication.
// Forwards function func to funcRef, funcNeon or funcCl, depending on the value of backendId.
#define FORWARD_LAYER_SUPPORT_FUNC(backendId, func, ...) \
    std::string reasonIfUnsupportedFull; \
    bool isSupported; \
    try { \
        auto const& backendRegistry = BackendRegistryInstance(); \
        if (!backendRegistry.IsBackendRegistered(backendId)) \
        { \
            std::stringstream ss; \
            ss << __func__ << " is not supported on " << backendId << " because this backend is not registered."; \
            reasonIfUnsupportedFull = ss.str(); \
            isSupported = false; \
        } \
        else \
        { \
            auto factoryFunc = backendRegistry.GetFactory(backendId); \
            auto backendObject = factoryFunc(); \
            auto layerSupportObject = backendObject->GetLayerSupport(); \
            isSupported = layerSupportObject->func(__VA_ARGS__, Optional<std::string&>(reasonIfUnsupportedFull)); \
            CopyErrorMessage(reasonIfUnsupported, reasonIfUnsupportedFull.c_str(), reasonIfUnsupportedMaxLength); \
        } \
    } catch (const InvalidArgumentException &e) { \
        /* re-throwing with more context information */ \
        throw InvalidArgumentException(e, "Failed to check layer support", CHECK_LOCATION()); \
    } \
    return isSupported;

bool CheckTensorDataTypesEqual(const TensorInfo& input0, const TensorInfo& input1)
{
    return input0.GetDataType() == input1.GetDataType();
}

using TensorInfos = std::vector<TensorInfo>;

bool IsActivationSupported(const BackendId& backend,
                           const TensorInfo& input,
                           const TensorInfo& output,
                           const ActivationDescriptor& descriptor,
                           char* reasonIfUnsupported,
                           size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Activation,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsAdditionSupported(const BackendId& backend,
                         const TensorInfo& input0,
                         const TensorInfo& input1,
                         const TensorInfo& output,
                         char* reasonIfUnsupported,
                         size_t reasonIfUnsupportedMaxLength)
{
    if(!CheckTensorDataTypesEqual(input0, input1))
    {
        return false;
    }

    TensorInfos infos{input0, input1, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Addition,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsArgMinMaxSupported(const BackendId& backend,
                          const TensorInfo& input,
                          const TensorInfo& output,
                          const ArgMinMaxDescriptor& descriptor,
                          char* reasonIfUnsupported,
                          size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::ArgMinMax,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsBatchNormalizationSupported(const BackendId& backend,
                                   const TensorInfo& input,
                                   const TensorInfo& output,
                                   const TensorInfo& mean,
                                   const TensorInfo& var,
                                   const TensorInfo& beta,
                                   const TensorInfo& gamma,
                                   const BatchNormalizationDescriptor& descriptor,
                                   char* reasonIfUnsupported,
                                   size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output, mean, var, beta, gamma};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::BatchNormalization,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsBatchToSpaceNdSupported(const BackendId& backend,
                               const TensorInfo& input,
                               const TensorInfo& output,
                               const BatchToSpaceNdDescriptor& descriptor,
                               char* reasonIfUnsupported,
                               size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::BatchToSpaceNd,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsConcatSupported(const BackendId& backend,
                       std::vector<const TensorInfo*> inputs,
                       const TensorInfo& output,
                       const OriginsDescriptor& descriptor,
                       char* reasonIfUnsupported,
                       size_t reasonIfUnsupportedMaxLength)
{
    ARMNN_ASSERT(inputs.size() > 0);

    TensorInfos infos;
    for (const TensorInfo* inputInfo : inputs)
    {
        infos.push_back(*inputInfo);
    }
    infos.push_back(output);

    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Concat,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsConstantSupported(const BackendId& backend,
                         const TensorInfo& output,
                         char* reasonIfUnsupported,
                         size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Constant,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsConvertFp16ToFp32Supported(const BackendId& backend,
                                  const TensorInfo& input,
                                  const TensorInfo& output,
                                  char* reasonIfUnsupported,
                                  size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::ConvertFp16ToFp32,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsConvertFp32ToFp16Supported(const BackendId& backend,
                                  const TensorInfo& input,
                                  const TensorInfo& output,
                                  char* reasonIfUnsupported,
                                  size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::ConvertFp32ToFp16,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsConvolution2dSupported(const BackendId& backend,
                              const TensorInfo& input,
                              const TensorInfo& output,
                              const Convolution2dDescriptor& descriptor,
                              const TensorInfo& weights,
                              const Optional<TensorInfo>& biases,
                              char* reasonIfUnsupported,
                              size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output, weights, biases.value()};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Convolution2d,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsDebugSupported(const BackendId& backend,
                      const TensorInfo& input,
                      const TensorInfo& output,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Debug,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsDepthwiseConvolutionSupported(const BackendId& backend,
                                     const TensorInfo& input,
                                     const TensorInfo& output,
                                     const DepthwiseConvolution2dDescriptor& descriptor,
                                     const TensorInfo& weights,
                                     const Optional<TensorInfo>& biases,
                                     char* reasonIfUnsupported,
                                     size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output, weights, biases.value()};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::DepthwiseConvolution2d,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsDequantizeSupported(const BackendId& backend,
                           const TensorInfo& input,
                           const TensorInfo& output,
                           char* reasonIfUnsupported,
                           size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Dequantize,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsDetectionPostProcessSupported(const BackendId& backend,
                                     const TensorInfo& input0,
                                     const TensorInfo& input1,
                                     const DetectionPostProcessDescriptor& descriptor,
                                     char* reasonIfUnsupported,
                                     size_t reasonIfUnsupportedMaxLength);

bool IsDivisionSupported(const BackendId& backend,
                         const TensorInfo& input0,
                         const TensorInfo& input1,
                         const TensorInfo& output,
                         char* reasonIfUnsupported,
                         size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input0, input1, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Division,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsEqualSupported(const BackendId& backend,
                      const TensorInfo& input0,
                      const TensorInfo& input1,
                      const TensorInfo& output,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input0, input1, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Comparison,
                               infos,
                               ComparisonDescriptor(ComparisonOperation::Equal),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsFakeQuantizationSupported(const BackendId& backend,
                                 const TensorInfo& input,
                                 const FakeQuantizationDescriptor& descriptor,
                                 char* reasonIfUnsupported,
                                 size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::FakeQuantization,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsFloorSupported(const BackendId& backend,
                      const TensorInfo& input,
                      const TensorInfo& output,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    // By definition (that is, regardless of compute device), shapes and data type must match.
    if (input.GetShape() != output.GetShape() || input.GetDataType() != output.GetDataType())
    {
        return false;
    }

    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Floor,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsFullyConnectedSupported(const BackendId& backend,
                               const TensorInfo& input,
                               const TensorInfo& output,
                               const TensorInfo& weights,
                               const TensorInfo& biases,
                               const FullyConnectedDescriptor& descriptor,
                               char* reasonIfUnsupported,
                               size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output, weights, biases};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::FullyConnected,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsGatherSupported(const BackendId& backend,
                       const TensorInfo& input0,
                       const TensorInfo& input1,
                       const TensorInfo& output,
                       const GatherDescriptor& descriptor,
                       char* reasonIfUnsupported,
                       size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input0, input1, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Gather,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsGreaterSupported(const BackendId& backend,
                        const TensorInfo& input0,
                        const TensorInfo& input1,
                        const TensorInfo& output,
                        char* reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input0, input1, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Comparison,
                               infos,
                               ComparisonDescriptor(ComparisonOperation::Greater),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsInputSupported(const BackendId& backend,
                      const TensorInfo& input,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Input,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}


bool IsL2NormalizationSupported(const BackendId& backend,
                                const TensorInfo& input,
                                const TensorInfo& output,
                                const L2NormalizationDescriptor& descriptor,
                                char* reasonIfUnsupported,
                                size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::L2Normalization,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsLstmSupported(const BackendId& backend, const TensorInfo& input, const TensorInfo& outputStateIn,
                     const TensorInfo& cellStateIn, const TensorInfo& scratchBuffer,
                     const TensorInfo& outputStateOut, const TensorInfo& cellStateOut,
                     const TensorInfo& output, const LstmDescriptor& descriptor,
                     const LstmInputParamsInfo& paramsInfo, char* reasonIfUnsupported,
                     size_t reasonIfUnsupportedMaxLength)

{
    TensorInfos infos{input, outputStateIn, cellStateIn, scratchBuffer, outputStateOut, cellStateOut, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Lstm,
                               infos,
                               descriptor,
                               paramsInfo,
                               EmptyOptional());
}

bool IsMaximumSupported(const BackendId& backend,
                        const TensorInfo& input0,
                        const TensorInfo& input1,
                        const TensorInfo& output,
                        char* reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input0, input1, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Maximum,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsMeanSupported(const BackendId& backend,
                     const TensorInfo& input,
                     const TensorInfo& output,
                     const MeanDescriptor& descriptor,
                     char* reasonIfUnsupported,
                     size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Mean,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsMemCopySupported(const BackendId &backend,
                        const TensorInfo &input,
                        const TensorInfo &output,
                        char *reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::MemCopy,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsMemImportSupported(const BackendId &backend,
                          const TensorInfo &input,
                          const TensorInfo &output,
                          char *reasonIfUnsupported,
                          size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::MemImport,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsMergeSupported(const BackendId& backend,
                      const TensorInfo& input0,
                      const TensorInfo& input1,
                      const TensorInfo& output,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input0, input1, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Merge,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsMinimumSupported(const BackendId& backend,
                        const TensorInfo& input0,
                        const TensorInfo& input1,
                        const TensorInfo& output,
                        char* reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input0, input1, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Minimum,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsMultiplicationSupported(const BackendId& backend,
                               const TensorInfo& input0,
                               const TensorInfo& input1,
                               const TensorInfo& output,
                               char* reasonIfUnsupported,
                               size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input0, input1, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Multiplication,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsNormalizationSupported(const BackendId& backend,
                              const TensorInfo& input,
                              const TensorInfo& output,
                              const NormalizationDescriptor& descriptor,
                              char* reasonIfUnsupported,
                              size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Normalization,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsOutputSupported(const BackendId& backend,
                       const TensorInfo& output,
                       char* reasonIfUnsupported,
                       size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Output,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());;
}

bool IsPadSupported(const BackendId& backend,
                    const TensorInfo& input,
                    const TensorInfo& output,
                    const PadDescriptor& descriptor,
                    char* reasonIfUnsupported,
                    size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Pad,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsQuantizeSupported(const BackendId& backend,
                         const TensorInfo& input,
                         const TensorInfo& output,
                         char* reasonIfUnsupported,
                         size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Quantize,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsQLstmSupported(const BackendId& backend,
                      const TensorInfo& input,
                      const TensorInfo& previousOutputIn,
                      const TensorInfo& previousCellStateIn,
                      const TensorInfo& outputStateOut,
                      const TensorInfo& cellStateOut,
                      const TensorInfo& output,
                      const QLstmDescriptor& descriptor,
                      const LstmInputParamsInfo& paramsInfo,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)

{
    TensorInfos infos{input, previousOutputIn, previousCellStateIn, outputStateOut, cellStateOut, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::QLstm,
                               infos,
                               descriptor,
                               paramsInfo,
                               EmptyOptional());
}

bool IsQuantizedLstmSupported(const BackendId& backend,
                              const TensorInfo& input,
                              const TensorInfo& previousCellStateIn,
                              const TensorInfo& previousOutputIn,
                              const TensorInfo& cellStateOut,
                              const TensorInfo& output,
                              const QuantizedLstmInputParamsInfo& paramsInfo,
                              char* reasonIfUnsupported,
                              size_t reasonIfUnsupportedMaxLength)

{
    TensorInfos infos{input, previousCellStateIn, previousOutputIn, cellStateOut, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::QuantizedLstm,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               paramsInfo);
}


bool IsPermuteSupported(const BackendId& backend,
                        const TensorInfo& input,
                        const TensorInfo& output,
                        const PermuteDescriptor& descriptor,
                        char* reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Permute,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsPooling2dSupported(const BackendId& backend,
                          const TensorInfo& input,
                          const TensorInfo& output,
                          const Pooling2dDescriptor& descriptor,
                          char* reasonIfUnsupported,
                          size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Pooling2d,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsPreluSupported(const BackendId& backend,
                      const TensorInfo& input,
                      const TensorInfo& alpha,
                      const TensorInfo& output,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, alpha, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Prelu,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsReduceSupported(const BackendId& backend,
                       const TensorInfo& input,
                       const TensorInfo& output,
                       const ReduceDescriptor& descriptor,
                       char* reasonIfUnsupported,
                       size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Reduce,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsReshapeSupported(const BackendId& backend,
                        const TensorInfo& input,
                        const TensorInfo& output,
                        const ReshapeDescriptor& descriptor,
                        char* reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Reshape,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsResizeSupported(const BackendId& backend,
                       const TensorInfo& input,
                       const TensorInfo& output,
                       const ResizeDescriptor& descriptor,
                       char* reasonIfUnsupported,
                       size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Resize,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsSoftmaxSupported(const BackendId& backend,
                        const TensorInfo& input,
                        const TensorInfo& output,
                        const SoftmaxDescriptor& descriptor,
                        char* reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Softmax,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsSpaceToBatchNdSupported(const BackendId& backend,
                               const TensorInfo& input,
                               const TensorInfo& output,
                               const SpaceToBatchNdDescriptor& descriptor,
                               char* reasonIfUnsupported,
                               size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::SpaceToBatchNd,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsSpaceToDepthSupported(const BackendId& backend,
                             const TensorInfo& input,
                             const TensorInfo& output,
                             const SpaceToDepthDescriptor& descriptor,
                             char* reasonIfUnsupported,
                             size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::SpaceToDepth,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsSplitterSupported(const BackendId& backend,
                         const TensorInfo& input,
                         const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                         const ViewsDescriptor& descriptor,
                         char* reasonIfUnsupported,
                         size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input};
    for (TensorInfo outInfo : outputs)
    {
        infos.push_back(outInfo);
    }

    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Splitter,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsStridedSliceSupported(const BackendId& backend,
                             const TensorInfo& input,
                             const TensorInfo& output,
                             const StridedSliceDescriptor& descriptor,
                             char* reasonIfUnsupported,
                             size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::StridedSlice,
                               infos,
                               descriptor,
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsSubtractionSupported(const BackendId& backend,
                            const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            char* reasonIfUnsupported,
                            size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input0, input1, output};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Subtraction,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

bool IsSwitchSupported(const BackendId& backend,
                       const TensorInfo& input0,
                       const TensorInfo& input1,
                       const TensorInfo& output0,
                       const TensorInfo& output1,
                       char* reasonIfUnsupported,
                       size_t reasonIfUnsupportedMaxLength)
{
    TensorInfos infos{input0, input1, output0, output1};
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsLayerSupported,
                               LayerType::Switch,
                               infos,
                               BaseDescriptor(),
                               EmptyOptional(),
                               EmptyOptional());
}

} // namespace armnn
