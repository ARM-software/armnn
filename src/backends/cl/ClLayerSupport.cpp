//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClLayerSupport.hpp"
#include "ClBackendId.hpp"
#include "ClBackendModelContext.hpp"

#include <armnn/BackendRegistry.hpp>

#include <InternalTypes.hpp>
#include <LayerSupportCommon.hpp>

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#if defined(ARMCOMPUTECL_ENABLED)
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include "workloads/ClAbsWorkload.hpp"
#include "workloads/ClAdditionWorkload.hpp"
#include "workloads/ClActivationWorkload.hpp"
#include "workloads/ClArgMinMaxWorkload.hpp"
#include "workloads/ClBatchNormalizationFloatWorkload.hpp"
#include "workloads/ClBatchToSpaceNdWorkload.hpp"
#include "workloads/ClCastWorkload.hpp"
#include "workloads/ClChannelShuffleWorkload.hpp"
#include "workloads/ClComparisonWorkload.hpp"
#include "workloads/ClConstantWorkload.hpp"
#include "workloads/ClConvertFp16ToFp32Workload.hpp"
#include "workloads/ClConvertFp32ToFp16Workload.hpp"
#include "workloads/ClConvolution2dWorkload.hpp"
#include "workloads/ClConvolution3dWorkload.hpp"
#include "workloads/ClDepthToSpaceWorkload.hpp"
#include "workloads/ClDepthwiseConvolutionWorkload.hpp"
#include "workloads/ClDequantizeWorkload.hpp"
#include "workloads/ClDivisionWorkload.hpp"
#include "workloads/ClExpWorkload.hpp"
#include "workloads/ClFillWorkload.hpp"
#include "workloads/ClFloorFloatWorkload.hpp"
#include "workloads/ClFullyConnectedWorkload.hpp"
#include "workloads/ClGatherWorkload.hpp"
#include "workloads/ClInstanceNormalizationWorkload.hpp"
#include "workloads/ClL2NormalizationFloatWorkload.hpp"
#include "workloads/ClLogWorkload.hpp"
#include "workloads/ClLogSoftmaxWorkload.hpp"
#include "workloads/ClLogicalAndWorkload.hpp"
#include "workloads/ClLogicalNotWorkload.hpp"
#include "workloads/ClLogicalOrWorkload.hpp"
#include "workloads/ClLstmFloatWorkload.hpp"
#include "workloads/ClMaximumWorkload.hpp"
#include "workloads/ClMeanWorkload.hpp"
#include "workloads/ClConcatWorkload.hpp"
#include "workloads/ClMinimumWorkload.hpp"
#include "workloads/ClMultiplicationWorkload.hpp"
#include "workloads/ClNegWorkload.hpp"
#include "workloads/ClNormalizationFloatWorkload.hpp"
#include "workloads/ClPadWorkload.hpp"
#include "workloads/ClPermuteWorkload.hpp"
#include "workloads/ClPooling2dWorkload.hpp"
#include "workloads/ClPreluWorkload.hpp"
#include "workloads/ClQLstmWorkload.hpp"
#include "workloads/ClQuantizedLstmWorkload.hpp"
#include "workloads/ClQuantizeWorkload.hpp"
#include "workloads/ClReduceWorkload.hpp"
#include "workloads/ClReshapeWorkload.hpp"
#include "workloads/ClResizeWorkload.hpp"
#include "workloads/ClRsqrtWorkload.hpp"
#include "workloads/ClSinWorkload.hpp"
#include "workloads/ClSliceWorkload.hpp"
#include "workloads/ClSoftmaxWorkload.hpp"
#include "workloads/ClSpaceToBatchNdWorkload.hpp"
#include "workloads/ClSpaceToDepthWorkload.hpp"
#include "workloads/ClSplitterWorkload.hpp"
#include "workloads/ClStackWorkload.hpp"
#include "workloads/ClStridedSliceWorkload.hpp"
#include "workloads/ClSubtractionWorkload.hpp"
#include "workloads/ClTransposeConvolution2dWorkload.hpp"
#include "workloads/ClTransposeWorkload.hpp"
#endif


namespace armnn
{

namespace
{

template<unsigned int FilterSize>
bool IsMatchingSize2d(const TensorInfo& weightInfo)
{
    // Width & Height must match.
    return (weightInfo.GetShape()[3] == FilterSize) && (weightInfo.GetShape()[2] == FilterSize);
}

template<uint32_t ValidStride>
bool IsMatchingStride(uint32_t actualStride)
{
    return ValidStride == actualStride;
}

template<uint32_t FirstStride, uint32_t SecondStride, uint32_t... ValidStrides>
bool IsMatchingStride(uint32_t actualStride)
{
    return IsMatchingStride<FirstStride>(actualStride) || IsMatchingStride<SecondStride, ValidStrides...>(actualStride);
}

template<typename ... Args>
bool IsClBackendSupported(Optional<std::string&> reasonIfUnsupported, Args... args)
{
    IgnoreUnused(reasonIfUnsupported, (args)...);
#if defined(ARMCOMPUTECL_ENABLED)
    return true;
#else
    if (reasonIfUnsupported)
    {
        reasonIfUnsupported.value() = "The armnn library has been built without CL support";
    }
    return false;
#endif
}

#if defined(ARMCOMPUTECL_ENABLED)
#define FORWARD_CL_LAYER_SUPPORT_FUNC(expr) (expr)
#else
#define FORWARD_CL_LAYER_SUPPORT_FUNC(expr) IsClBackendSupported(reasonIfUnsupported)
#endif

#if defined(ARMCOMPUTECL_ENABLED)
template<class FuncType, class... Args>
inline bool IsWorkloadSupported(FuncType&& func, Optional<std::string&> reasonIfUnsupported, Args&&... args)
{
    arm_compute::Status aclStatus = func(std::forward<Args>(args)...);
    const bool supported = (aclStatus.error_code() == arm_compute::ErrorCode::OK);
    if (!supported && reasonIfUnsupported)
    {
        reasonIfUnsupported.value() = aclStatus.error_description();
    }
    return supported;
}

#define FORWARD_WORKLOAD_VALIDATE_FUNC(func, reasonIfUnsupported, ...) \
    return IsWorkloadSupported(func, reasonIfUnsupported, __VA_ARGS__);
#else
#define FORWARD_WORKLOAD_VALIDATE_FUNC(func, reasonIfUnsupported, ...) \
    return IsClBackendSupported(reasonIfUnsupported, __VA_ARGS__);
#endif

template<typename FloatFunc, typename Uint8Func, typename ... Params>
bool IsSupportedForDataTypeCl(Optional<std::string&> reasonIfUnsupported,
                              DataType dataType,
                              FloatFunc floatFuncPtr,
                              Uint8Func uint8FuncPtr,
                              Params&&... params)
{
    return IsClBackendSupported(reasonIfUnsupported) &&
        IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                      dataType,
                                      floatFuncPtr,
                                      floatFuncPtr,
                                      uint8FuncPtr,
                                      &FalseFunc<>,
                                      &FalseFunc<>,
                                      std::forward<Params>(params)...);
}
} // anonymous namespace

ClLayerSupport::ClLayerSupport(const IBackendInternal::IBackendSpecificModelContextPtr& modelContextPtr)
    : m_ModelContextPtr(modelContextPtr)
{
}

ClLayerSupport::ClLayerSupport()
    : m_ModelContextPtr(nullptr)
{
}

bool ClLayerSupport::IsLayerSupported(const LayerType& type,
                                      const std::vector<TensorInfo>& infos,
                                      const BaseDescriptor& descriptor,
                                      const Optional<LstmInputParamsInfo>& lstmParamsInfo,
                                      const Optional<QuantizedLstmInputParamsInfo>& quantizedLstmParamsInfo,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    switch (type)
    {
        case LayerType::Activation:
            return IsActivationSupported(infos[0],
                                         infos[1],
                                         *(PolymorphicDowncast<const ActivationDescriptor*>(&descriptor)),
                                         reasonIfUnsupported);
        case LayerType::Addition:
            return IsAdditionSupported(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::ArgMinMax:
            return IsArgMinMaxSupported(infos[0],
                                        infos[1],
                                        *(PolymorphicDowncast<const ArgMinMaxDescriptor*>(&descriptor)),
                                        reasonIfUnsupported);
        case LayerType::BatchNormalization:
            return IsBatchNormalizationSupported(infos[0],
                                                 infos[1],
                                                 infos[2],
                                                 infos[3],
                                                 infos[4],
                                                 infos[5],
                                                 *(PolymorphicDowncast<const BatchNormalizationDescriptor*>
                                                     (&descriptor)),
                                                 reasonIfUnsupported);
        case LayerType::BatchToSpaceNd:
            return IsBatchToSpaceNdSupported(infos[0],
                                             infos[1],
                                             *(PolymorphicDowncast<const BatchToSpaceNdDescriptor*>(&descriptor)),
                                             reasonIfUnsupported);
        case LayerType::Comparison:
            return IsComparisonSupported(infos[0],
                                         infos[1],
                                         infos[2],
                                         *(PolymorphicDowncast<const ComparisonDescriptor*>(&descriptor)),
                                         reasonIfUnsupported);
        case LayerType::Concat:
        {
            std::vector<const TensorInfo*> inputInfos;
            for (uint32_t i = 0; i < (infos.size() - 1); i++)
            {
                inputInfos.push_back(&infos[i]);
            }
            return IsConcatSupported(inputInfos,
                                     infos[infos.size() - 1],
                                     *(PolymorphicDowncast<const OriginsDescriptor*>(&descriptor)),
                                     reasonIfUnsupported);
        }
        case LayerType::Constant:
            return IsConstantSupported(infos[0], reasonIfUnsupported);
        case LayerType::ConvertFp16ToFp32:
            return IsConvertFp16ToFp32Supported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::ConvertFp32ToFp16:
            return IsConvertFp32ToFp16Supported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::Convolution2d:
        {
            if (infos.size() != 4)
            {
                throw InvalidArgumentException("Invalid number of Convolution2d TensorInfos. "
                                               "TensorInfos should be of format: {input, output, weights, biases}.");
            }

            auto desc = *(PolymorphicDowncast<const Convolution2dDescriptor*>(&descriptor));
            if (infos[3] == TensorInfo())
            {
                return IsConvolution2dSupported(infos[0],
                                                infos[1],
                                                desc,
                                                infos[2],
                                                EmptyOptional(),
                                                reasonIfUnsupported);
            }
            else
            {
                return IsConvolution2dSupported(infos[0],
                                                infos[1],
                                                desc,
                                                infos[2],
                                                infos[3],
                                                reasonIfUnsupported);
            }
        }
        case LayerType::DepthToSpace:
            return IsDepthToSpaceSupported(infos[0],
                                           infos[1],
                                           *(PolymorphicDowncast<const DepthToSpaceDescriptor*>(&descriptor)),
                                           reasonIfUnsupported);
        case LayerType::DepthwiseConvolution2d:
        {
            if (infos.size() != 4)
            {
                throw InvalidArgumentException("Invalid number of DepthwiseConvolution2d TensorInfos. "
                                               "TensorInfos should be of format: {input, output, weights, biases}.");
            }

            auto desc = *(PolymorphicDowncast<const DepthwiseConvolution2dDescriptor*>(&descriptor));
            if (infos[3] == TensorInfo())
            {
                return IsDepthwiseConvolutionSupported(infos[0],
                                                infos[1],
                                                desc,
                                                infos[2],
                                                EmptyOptional(),
                                                reasonIfUnsupported);
            }
            else
            {
                return IsDepthwiseConvolutionSupported(infos[0],
                                                         infos[1],
                                                         desc,
                                                         infos[2],
                                                         infos[3],
                                                         reasonIfUnsupported);
            }
        }
        case LayerType::Dequantize:
            return IsDequantizeSupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::Division:
            return IsDivisionSupported(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::ElementwiseUnary:
            return IsElementwiseUnarySupported(infos[0],
                                               infos[1],
                                               *(PolymorphicDowncast<const ElementwiseUnaryDescriptor*>(&descriptor)),
                                               reasonIfUnsupported);
        case LayerType::Fill:
            return IsFillSupported(infos[0],
                                   infos[1],
                                   *(PolymorphicDowncast<const FillDescriptor*>(&descriptor)),
                                   reasonIfUnsupported);
        case LayerType::Floor:
            return IsFloorSupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::FullyConnected:
            return IsFullyConnectedSupported(infos[0],
                                             infos[1],
                                             infos[2],
                                             infos[3],
                                             *(PolymorphicDowncast<const FullyConnectedDescriptor*>(&descriptor)),
                                             reasonIfUnsupported);
        case LayerType::Gather:
            return IsGatherSupported(infos[0],
                                     infos[1],
                                     infos[2],
                                     *(PolymorphicDowncast<const GatherDescriptor*>(&descriptor)),
                                     reasonIfUnsupported);
        case LayerType::Input:
            return IsInputSupported(infos[0], reasonIfUnsupported);
        case LayerType::InstanceNormalization:
            return IsInstanceNormalizationSupported(infos[0],
                                                    infos[1],
                                                    *(PolymorphicDowncast<const InstanceNormalizationDescriptor*>
                                                        (&descriptor)),
                                                    reasonIfUnsupported);
        case LayerType::L2Normalization:
            return IsL2NormalizationSupported(infos[0],
                                              infos[1],
                                              *(PolymorphicDowncast<const L2NormalizationDescriptor*>(&descriptor)),
                                              reasonIfUnsupported);
        case LayerType::LogicalBinary:
            return IsLogicalBinarySupported(infos[0],
                                            infos[1],
                                            infos[2],
                                            *(PolymorphicDowncast<const LogicalBinaryDescriptor*>(&descriptor)),
                                            reasonIfUnsupported);
        case LayerType::LogSoftmax:
            return IsLogSoftmaxSupported(infos[0],
                                         infos[1],
                                         *(PolymorphicDowncast<const LogSoftmaxDescriptor*>(&descriptor)),
                                         reasonIfUnsupported);
        case LayerType::Lstm:
            return IsLstmSupported(infos[0],
                                   infos[1],
                                   infos[2],
                                   infos[3],
                                   infos[4],
                                   infos[5],
                                   infos[6],
                                   *(PolymorphicDowncast<const LstmDescriptor*>(&descriptor)),
                                   lstmParamsInfo.value(),
                                   reasonIfUnsupported);
        case LayerType::QLstm:
            return IsQLstmSupported(infos[0],
                                    infos[1],
                                    infos[2],
                                    infos[3],
                                    infos[4],
                                    infos[5],
                                    *(PolymorphicDowncast<const QLstmDescriptor*>(&descriptor)),
                                    lstmParamsInfo.value(),
                                    reasonIfUnsupported);
        case LayerType::Maximum:
            return IsMaximumSupported(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::Mean:
            return IsMeanSupported(infos[0],
                                   infos[1],
                                   *(PolymorphicDowncast<const MeanDescriptor*>(&descriptor)),
                                   reasonIfUnsupported);
        case LayerType::Minimum:
            return IsMinimumSupported(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::Multiplication:
            return IsMultiplicationSupported(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::Normalization:
            return IsNormalizationSupported(infos[0],
                                            infos[1],
                                            *(PolymorphicDowncast<const NormalizationDescriptor*>(&descriptor)),
                                            reasonIfUnsupported);
        case LayerType::Output:
            return IsOutputSupported(infos[0], reasonIfUnsupported);
        case LayerType::Pad:
            return IsPadSupported(infos[0],
                                  infos[1],
                                  *(PolymorphicDowncast<const PadDescriptor*>(&descriptor)),
                                  reasonIfUnsupported);
        case LayerType::Permute:
            return IsPermuteSupported(infos[0],
                                      infos[1],
                                      *(PolymorphicDowncast<const PermuteDescriptor*>(&descriptor)),
                                      reasonIfUnsupported);
        case LayerType::Pooling2d:
            return IsPooling2dSupported(infos[0],
                                        infos[1],
                                        *(PolymorphicDowncast<const Pooling2dDescriptor*>(&descriptor)),
                                        reasonIfUnsupported);
        case LayerType::Prelu:
            return IsPreluSupported(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::Quantize:
            return IsQuantizeSupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::QuantizedLstm:
            return IsQuantizedLstmSupported(infos[0],
                                            infos[1],
                                            infos[2],
                                            infos[3],
                                            infos[4],
                                            quantizedLstmParamsInfo.value(),
                                            reasonIfUnsupported);
        case LayerType::Reshape:
            return IsReshapeSupported(infos[0],
                                      infos[1],
                                      *(PolymorphicDowncast<const ReshapeDescriptor*>(&descriptor)),
                                      reasonIfUnsupported);
        case LayerType::Resize:
            return IsResizeSupported(infos[0],
                                     infos[1],
                                     *(PolymorphicDowncast<const ResizeDescriptor*>(&descriptor)),
                                     reasonIfUnsupported);
        case LayerType::Reduce:
            return IsReduceSupported(infos[0],
                                     infos[1],
                                     *(PolymorphicDowncast<const ReduceDescriptor*>(&descriptor)),
                                     reasonIfUnsupported);
        case LayerType::Slice:
            return IsSliceSupported(infos[0],
                                    infos[1],
                                    *(PolymorphicDowncast<const SliceDescriptor*>(&descriptor)),
                                    reasonIfUnsupported);
        case LayerType::Softmax:
            return IsSoftmaxSupported(infos[0],
                                      infos[1],
                                      *(PolymorphicDowncast<const SoftmaxDescriptor*>(&descriptor)),
                                      reasonIfUnsupported);
        case LayerType::SpaceToBatchNd:
            return IsSpaceToBatchNdSupported(infos[0],
                                             infos[1],
                                             *(PolymorphicDowncast<const SpaceToBatchNdDescriptor*>(&descriptor)),
                                             reasonIfUnsupported);
        case LayerType::SpaceToDepth:
            return IsSpaceToDepthSupported(infos[0],
                                           infos[1],
                                           *(PolymorphicDowncast<const SpaceToDepthDescriptor*>(&descriptor)),
                                           reasonIfUnsupported);
        case LayerType::Splitter:
        {
            std::vector<TensorInfo> outputInfos;
            for (uint32_t i = 1; i < infos.size(); i++)
            {
                outputInfos.push_back(infos[i]);
            }
            return IsSplitterSupported(infos[0],
                                       {outputInfos.begin(), outputInfos.end()},
                                       *(PolymorphicDowncast<const ViewsDescriptor*>(&descriptor)),
                                       reasonIfUnsupported);
        }
        case LayerType::Stack:
        {
            std::vector<const TensorInfo*> inputInfos;
            for (uint32_t i = 0; i < infos.size() - 1; i++)
            {
                inputInfos.push_back(&infos[i]);
            }
            return IsStackSupported(inputInfos,
                                    infos[infos.size() - 1],
                                    *(PolymorphicDowncast<const StackDescriptor*>(&descriptor)),
                                    reasonIfUnsupported);
        }
        case LayerType::StridedSlice:
            return IsStridedSliceSupported(infos[0],
                                           infos[1],
                                           *(PolymorphicDowncast<const StridedSliceDescriptor*>(&descriptor)),
                                           reasonIfUnsupported);
        case LayerType::Subtraction:
            return IsSubtractionSupported(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::Transpose:
            return IsTransposeSupported(infos[0],
                                        infos[1],
                                        *(PolymorphicDowncast<const TransposeDescriptor*>(&descriptor)),
                                        reasonIfUnsupported);
        case LayerType::TransposeConvolution2d:
        {
            if (infos.size() != 4)
            {
                throw InvalidArgumentException("Invalid number of TransposeConvolution2d TensorInfos. "
                                               "TensorInfos should be of format: {input, output, weights, biases}.");
            }

            auto desc = *(PolymorphicDowncast<const TransposeConvolution2dDescriptor*>(&descriptor));
            if (infos[3] == TensorInfo())
            {
                return IsTransposeConvolution2dSupported(infos[0],
                                                         infos[1],
                                                         desc,
                                                         infos[2],
                                                         EmptyOptional(),
                                                         reasonIfUnsupported);
            }
            else
            {
                return IsTransposeConvolution2dSupported(infos[0],
                                                         infos[1],
                                                         desc,
                                                         infos[2],
                                                         infos[3],
                                                         reasonIfUnsupported);
            }
        }
        case LayerType::Cast:
            return IsCastSupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::ChannelShuffle:
            return IsChannelShuffleSupported(infos[0],
                                             infos[1],
                                             *(PolymorphicDowncast<const ChannelShuffleDescriptor*>(&descriptor)),
                                             reasonIfUnsupported);
        case LayerType::Convolution3d:
        {
            if (infos.size() != 4)
            {
                throw InvalidArgumentException("Invalid number of Convolution3d TensorInfos. "
                                               "TensorInfos should be of format: {input, output, weights, biases}.");
            }

            auto desc = *(PolymorphicDowncast<const Convolution3dDescriptor*>(&descriptor));
            if (infos[3] == TensorInfo())
            {
                return IsConvolution3dSupported(infos[0],
                                                infos[1],
                                                desc,
                                                infos[2],
                                                EmptyOptional(),
                                                reasonIfUnsupported);
            }
            else
            {
                return IsConvolution3dSupported(infos[0],
                                                infos[1],
                                                desc,
                                                infos[2],
                                                infos[3],
                                                reasonIfUnsupported);
            }
        }
        case LayerType::MemCopy:
            return LayerSupportBase::IsMemCopySupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::MemImport:
            return LayerSupportBase::IsMemImportSupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::Map:
            return true;
        case LayerType::Unmap:
            return true;
        case LayerType::Merge:
            return LayerSupportBase::IsMergeSupported(infos[0],
                                                      infos[1],
                                                      infos[2],
                                                      reasonIfUnsupported);
        case LayerType::Rank:
            return true;
        case LayerType::Shape:
            return LayerSupportBase::IsShapeSupported(infos[0],
                                                      infos[1],
                                                      reasonIfUnsupported);
        case LayerType::ConvertBf16ToFp32:
            return LayerSupportBase::IsConvertBf16ToFp32Supported(infos[0],
                                                                  infos[1],
                                                                  reasonIfUnsupported);
        case LayerType::ConvertFp32ToBf16:
            return LayerSupportBase::IsConvertFp32ToBf16Supported(infos[0],
                                                                  infos[1],
                                                                  reasonIfUnsupported);
        default:
            // layers not supported in cl by default:
            // debug, detectionpostprocess, fakequantization, precompiled,
            // standin, switch, unidirectionalsequencelstm, pooling3d
            return false;
    }
}

bool ClLayerSupport::IsActivationSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const ActivationDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClActivationWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool ClLayerSupport::IsAdditionSupported(const TensorInfo& input0,
                                         const TensorInfo& input1,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClAdditionValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output,
                                   nullptr);
}

bool ClLayerSupport::IsArgMinMaxSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const ArgMinMaxDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{

    FORWARD_WORKLOAD_VALIDATE_FUNC(ClArgMinMaxWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool ClLayerSupport::IsBatchNormalizationSupported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const TensorInfo& mean,
                                                   const TensorInfo& var,
                                                   const TensorInfo& beta,
                                                   const TensorInfo& gamma,
                                                   const BatchNormalizationDescriptor& descriptor,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClBatchNormalizationValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   mean,
                                   var,
                                   beta,
                                   gamma,
                                   descriptor,
                                   nullptr);
}

bool ClLayerSupport::IsBatchToSpaceNdSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const BatchToSpaceNdDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClBatchToSpaceNdWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool ClLayerSupport::IsCastSupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClCastValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output);
}

bool ClLayerSupport::IsChannelShuffleSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const ChannelShuffleDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClChannelShuffleValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool ClLayerSupport::IsComparisonSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output,
                                           const ComparisonDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClComparisonWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output,
                                   descriptor);
}

bool ClLayerSupport::IsConcatSupported(const std::vector<const TensorInfo*> inputs,
                                       const TensorInfo& output,
                                       const OriginsDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    if (descriptor.GetNumDimensions() <= descriptor.GetConcatAxis())
    {
        SetValueChecked(reasonIfUnsupported, "Cl Concat: Concat axis > Number of dimensions.");
        return false;
    }

    unsigned int concatInnerAxis = (descriptor.GetNumDimensions() - descriptor.GetConcatAxis()) - 1;
    if(concatInnerAxis < 3) // Width, height, or channels
    {
        FORWARD_WORKLOAD_VALIDATE_FUNC(ClConcatWorkloadValidate,
                                       reasonIfUnsupported,
                                       inputs,
                                       output,
                                       descriptor);
    }
    else if (concatInnerAxis == 3)
    {
        // We rely on the sub-tensor optimization to handle the batch dimension for 4D tensors. If we can't use
        // sub-tensors for this then we can't support it. Here is where we check that the sub-tensors will work.
        for (auto& input : inputs)
        {
            if (input && !output.IsTypeSpaceMatch(*input)) // Cannot use sub-tensors if the types are not same space
            {
                SetValueChecked(reasonIfUnsupported, "Cl Concat: Types and quantization parameters must match.");
                return false;
            }
        }
        return true; // Sub-tensors support concat along batch
    }
    else // > 4 dimensions not supported.
    {
        SetValueChecked(reasonIfUnsupported, "Cl Concat: Maximum of 4 dimensions supported.");
        return false;
    }
}

bool ClLayerSupport::IsConstantSupported(const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClConstantWorkloadValidate,
                                   reasonIfUnsupported,
                                   output);
}

bool ClLayerSupport::IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClConvertFp16ToFp32WorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output);
}

bool ClLayerSupport::IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClConvertFp32ToFp16WorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output);
}

bool ClLayerSupport::IsConvolution2dSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const Convolution2dDescriptor& descriptor,
                                              const TensorInfo& weights,
                                              const Optional<TensorInfo>& biases,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    bool isFastMathEnabled = false;
#if defined(ARMCOMPUTECL_ENABLED)
    if (m_ModelContextPtr)
    {
        if (m_ModelContextPtr.get() != nullptr)
        {
            auto modelOptions = dynamic_cast<ClBackendModelContext*>(m_ModelContextPtr.get());
            if (modelOptions)
            {
                isFastMathEnabled = modelOptions->IsFastMathEnabled();
            }
        }
    }
#endif

    FORWARD_WORKLOAD_VALIDATE_FUNC(ClConvolution2dWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases,
                                   isFastMathEnabled,
                                   nullptr);
}

bool ClLayerSupport::IsConvolution3dSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const Convolution3dDescriptor& descriptor,
                                              const TensorInfo& weights,
                                              const Optional<TensorInfo>& biases,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    bool isFastMathEnabled = false;
#if defined(ARMCOMPUTECL_ENABLED)
    if (m_ModelContextPtr)
{
    if (m_ModelContextPtr.get() != nullptr)
    {
        auto modelOptions = dynamic_cast<ClBackendModelContext*>(m_ModelContextPtr.get());
        if (modelOptions)
        {
            isFastMathEnabled = modelOptions->IsFastMathEnabled();
        }
    }
}
#endif

    FORWARD_WORKLOAD_VALIDATE_FUNC(ClConvolution3dWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases,
                                   isFastMathEnabled,
                                   nullptr);
}

bool ClLayerSupport::IsDequantizeSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClDequantizeWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output);
}

bool ClLayerSupport::IsDepthToSpaceSupported(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const DepthToSpaceDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClDepthToSpaceWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool ClLayerSupport::IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const DepthwiseConvolution2dDescriptor& descriptor,
                                                     const TensorInfo& weights,
                                                     const Optional<TensorInfo>& biases,
                                                     Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClDepthwiseConvolutionWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases,
                                   nullptr);
}

bool ClLayerSupport::IsDilatedDepthwiseConvolutionSupported(const TensorInfo& input,
                                                            const TensorInfo& output,
                                                            const DepthwiseConvolution2dDescriptor& descriptor,
                                                            const TensorInfo& weights,
                                                            const Optional<TensorInfo>& biases,
                                                            Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClDepthwiseConvolutionWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases,
                                   nullptr);
}


bool ClLayerSupport::IsDivisionSupported(const TensorInfo& input0,
                                         const TensorInfo& input1,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClDivisionWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output,
                                   nullptr);
}

bool ClLayerSupport::IsElementwiseUnarySupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const ElementwiseUnaryDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    switch(descriptor.m_Operation)
    {
        case UnaryOperation::Abs:
            FORWARD_WORKLOAD_VALIDATE_FUNC(ClAbsWorkloadValidate,
                                           reasonIfUnsupported,
                                           input,
                                           output);
        case UnaryOperation::Exp:
            FORWARD_WORKLOAD_VALIDATE_FUNC(ClExpWorkloadValidate,
                                           reasonIfUnsupported,
                                           input,
                                           output);
        case UnaryOperation::Log:
            FORWARD_WORKLOAD_VALIDATE_FUNC(ClLogWorkloadValidate,
                                           reasonIfUnsupported,
                                           input,
                                           output);
        case UnaryOperation::LogicalNot:
            FORWARD_WORKLOAD_VALIDATE_FUNC(ClLogicalNotWorkloadValidate,
                                           reasonIfUnsupported,
                                           input,
                                           output);
        case UnaryOperation::Neg:
            FORWARD_WORKLOAD_VALIDATE_FUNC(ClNegWorkloadValidate,
                                           reasonIfUnsupported,
                                           input,
                                           output);
        case UnaryOperation::Rsqrt:
            FORWARD_WORKLOAD_VALIDATE_FUNC(ClRsqrtWorkloadValidate,
                                           reasonIfUnsupported,
                                           input,
                                           output);
        case UnaryOperation::Sin:
            FORWARD_WORKLOAD_VALIDATE_FUNC(ClSinWorkloadValidate,
                                           reasonIfUnsupported,
                                           input,
                                           output);
        default:
            return false;
    }
}

bool ClLayerSupport::IsFillSupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const FillDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    armnn::IgnoreUnused(input);
    armnn::IgnoreUnused(output);
    armnn::IgnoreUnused(descriptor);

    return IsClBackendSupported(reasonIfUnsupported);
}

bool ClLayerSupport::IsFloorSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClFloorWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output);
}

bool ClLayerSupport::IsFullyConnectedSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const TensorInfo& weights,
                                               const TensorInfo& biases,
                                               const FullyConnectedDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClFullyConnectedWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   weights,
                                   biases,
                                   descriptor,
                                   nullptr);
}

bool ClLayerSupport::IsGatherSupported(const TensorInfo& input0,
                                       const TensorInfo& input1,
                                       const TensorInfo& output,
                                       const GatherDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClGatherWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output,
                                   descriptor);
}

bool ClLayerSupport::IsInputSupported(const TensorInfo& input,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    return IsClBackendSupported(reasonIfUnsupported, input);
}

bool ClLayerSupport::IsInstanceNormalizationSupported(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const InstanceNormalizationDescriptor& descriptor,
                                                      Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClInstanceNormalizationWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool ClLayerSupport::IsL2NormalizationSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const L2NormalizationDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClL2NormalizationWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool ClLayerSupport::IsLogicalBinarySupported(const TensorInfo& input0,
                                              const TensorInfo& input1,
                                              const TensorInfo& output,
                                              const LogicalBinaryDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(output);

    switch(descriptor.m_Operation)
    {
        case LogicalBinaryOperation::LogicalAnd:
            FORWARD_WORKLOAD_VALIDATE_FUNC(ClLogicalAndWorkloadValidate,
                                           reasonIfUnsupported,
                                           input0,
                                           input1,
                                           output);
        case LogicalBinaryOperation::LogicalOr:
            FORWARD_WORKLOAD_VALIDATE_FUNC(ClLogicalOrWorkloadValidate,
                                           reasonIfUnsupported,
                                           input0,
                                           input1,
                                           output);
        default:
            return false;
    }
}


bool ClLayerSupport::IsLogSoftmaxSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const LogSoftmaxDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClLogSoftmaxWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool ClLayerSupport::IsLstmSupported(const TensorInfo& input,
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
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClLstmFloatWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   outputStateIn,
                                   cellStateIn,
                                   scratchBuffer,
                                   outputStateOut,
                                   cellStateOut,
                                   output,
                                   descriptor,
                                   paramsInfo);
}

bool ClLayerSupport::IsMaximumSupported(const TensorInfo& input0,
                                        const TensorInfo& input1,
                                        const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClMaximumWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output);
}

bool ClLayerSupport::IsMeanSupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const MeanDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClMeanValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool ClLayerSupport::IsMinimumSupported(const TensorInfo& input0,
                                        const TensorInfo& input1,
                                        const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClMinimumWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output);
}

bool ClLayerSupport::IsMultiplicationSupported(const TensorInfo& input0,
                                               const TensorInfo& input1,
                                               const TensorInfo& output,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClMultiplicationWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output,
                                   nullptr);
}

bool ClLayerSupport::IsNormalizationSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const NormalizationDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClNormalizationWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool ClLayerSupport::IsOutputSupported(const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return IsClBackendSupported(reasonIfUnsupported, output);
}

bool ClLayerSupport::IsPadSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const PadDescriptor& descriptor,
                                    Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClPadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool ClLayerSupport::IsPermuteSupported(const TensorInfo& input,
                                        const TensorInfo& output,
                                        const PermuteDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClPermuteWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool ClLayerSupport::IsPooling2dSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const Pooling2dDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClPooling2dWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool ClLayerSupport::IsPreluSupported(const armnn::TensorInfo &input,
                                      const armnn::TensorInfo &alpha,
                                      const armnn::TensorInfo &output,
                                      armnn::Optional<std::string &> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClPreluWorkloadValidate, reasonIfUnsupported, input, alpha, output);
}

bool ClLayerSupport::IsQLstmSupported(const TensorInfo& input,
                                      const TensorInfo& previousOutputIn,
                                      const TensorInfo& previousCellStateIn,
                                      const TensorInfo& outputStateOut,
                                      const TensorInfo& cellStateOut,
                                      const TensorInfo& output,
                                      const QLstmDescriptor& descriptor,
                                      const LstmInputParamsInfo& paramsInfo,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    if (input.GetDataType()               == armnn::DataType::QAsymmS8 &&
        previousOutputIn.GetDataType()    == armnn::DataType::QAsymmS8 &&
        previousCellStateIn.GetDataType() == armnn::DataType::QSymmS16 &&
        outputStateOut.GetDataType()      == armnn::DataType::QAsymmS8 &&
        cellStateOut.GetDataType()        == armnn::DataType::QSymmS16 &&
        output.GetDataType()              == armnn::DataType::QAsymmS8)
    {
        FORWARD_WORKLOAD_VALIDATE_FUNC(ClQLstmWorkloadValidate,
                                       reasonIfUnsupported,
                                       input,
                                       previousCellStateIn,
                                       previousOutputIn,
                                       cellStateOut,
                                       outputStateOut,
                                       output,
                                       descriptor,
                                       paramsInfo);
    }
    else
    {
        return false;
    }
}

bool ClLayerSupport::IsQuantizedLstmSupported(const TensorInfo& input,
                                              const TensorInfo& previousCellStateIn,
                                              const TensorInfo& previousOutputIn,
                                              const TensorInfo& cellStateOut,
                                              const TensorInfo& output,
                                              const QuantizedLstmInputParamsInfo& paramsInfo,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClQuantizedLstmWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   previousCellStateIn,
                                   previousOutputIn,
                                   cellStateOut,
                                   output,
                                   paramsInfo);
}

bool ClLayerSupport::IsQuantizeSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClQuantizeWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output);
}

bool ClLayerSupport::IsReduceSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const ReduceDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClReduceWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool ClLayerSupport::IsReshapeSupported(const TensorInfo& input,
                                        const TensorInfo& output,
                                        const ReshapeDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClReshapeWorkloadValidate, reasonIfUnsupported, input, output);
}

bool ClLayerSupport::IsResizeSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const ResizeDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClResizeWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool ClLayerSupport::IsSliceSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const SliceDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClSliceWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool ClLayerSupport::IsSoftmaxSupported(const TensorInfo& input,
                                        const TensorInfo& output,
                                        const SoftmaxDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClSoftmaxWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool ClLayerSupport::IsSpaceToBatchNdSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const SpaceToBatchNdDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClSpaceToBatchNdWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool ClLayerSupport::IsSpaceToDepthSupported(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const SpaceToDepthDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClSpaceToDepthWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool ClLayerSupport::IsSplitterSupported(const TensorInfo& input,
                                         const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                                         const ViewsDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
#if defined(ARMCOMPUTECL_ENABLED)
    // Split along the last dimension, cannot use sub-tensors
    // as width and height of the sub-tensors do not match
    // the width and height of the parent tensor
    // in case of input with more than 2D.
    std::set<unsigned int> splitAxis = ComputeSplitAxis(descriptor, input.GetShape());
    if (descriptor.GetNumDimensions() > 2 && splitAxis.size() == 1 &&
        *splitAxis.begin() == descriptor.GetNumDimensions() - 1 )
    {
        FORWARD_WORKLOAD_VALIDATE_FUNC(ClSplitterWorkloadValidate,
                                       reasonIfUnsupported,
                                       input,
                                       outputs,
                                       *splitAxis.begin());
    }
#endif
    IgnoreUnused(descriptor);
    for (auto output : outputs)
    {
        if (!input.IsTypeSpaceMatch(output)) // Cannot use sub-tensors if the types are not same space
        {
            SetValueChecked(reasonIfUnsupported, "Cl Splitter: Types and quantization parameters must match.");
            return false;
        }
    }
    return true;
}

bool ClLayerSupport::IsStackSupported(const std::vector<const TensorInfo*>& inputs,
                                      const TensorInfo& output,
                                      const StackDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClStackWorkloadValidate,
                                   reasonIfUnsupported,
                                   inputs,
                                   output,
                                   descriptor);
}

bool ClLayerSupport::IsStridedSliceSupported(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const StridedSliceDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClStridedSliceWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool ClLayerSupport::IsSubtractionSupported(const TensorInfo& input0,
                                            const TensorInfo& input1,
                                            const TensorInfo& output,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClSubtractionValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output,
                                   nullptr);
}

bool ClLayerSupport::IsTransposeConvolution2dSupported(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const TransposeConvolution2dDescriptor& descriptor,
                                                       const TensorInfo& weights,
                                                       const Optional<TensorInfo>& biases,
                                                       Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClTransposeConvolution2dWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases);
}

bool ClLayerSupport::IsTransposeSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const TransposeDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClTransposeWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

} // namespace armnn
