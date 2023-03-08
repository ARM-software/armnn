//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonLayerSupport.hpp"
#include "NeonBackendId.hpp"
#include "NeonBackendModelContext.hpp"

#include <armnn/Exceptions.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>
#include <armnn/BackendRegistry.hpp>

#include <InternalTypes.hpp>
#include <LayerSupportCommon.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#if defined(ARMCOMPUTENEON_ENABLED)
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include "workloads/NeonAbsWorkload.hpp"
#include "workloads/NeonAdditionWorkload.hpp"
#include "workloads/NeonActivationWorkload.hpp"
#include "workloads/NeonArgMinMaxWorkload.hpp"
#include "workloads/NeonBatchMatMulWorkload.hpp"
#include "workloads/NeonBatchNormalizationWorkload.hpp"
#include "workloads/NeonBatchToSpaceNdWorkload.hpp"
#include "workloads/NeonCastWorkload.hpp"
#include "workloads/NeonChannelShuffleWorkload.hpp"
#include "workloads/NeonComparisonWorkload.hpp"
#include "workloads/NeonConcatWorkload.hpp"
#include "workloads/NeonConstantWorkload.hpp"
#include "workloads/NeonConvolution2dWorkload.hpp"
#include "workloads/NeonConvolution3dWorkload.hpp"
#include "workloads/NeonDepthToSpaceWorkload.hpp"
#include "workloads/NeonDepthwiseConvolutionWorkload.hpp"
#include "workloads/NeonDequantizeWorkload.hpp"
#include "workloads/NeonExpWorkload.hpp"
#include "workloads/NeonInstanceNormalizationWorkload.hpp"
#include "workloads/NeonL2NormalizationFloatWorkload.hpp"
#include "workloads/NeonLogWorkload.hpp"
#include "workloads/NeonLogSoftmaxWorkload.hpp"
#include "workloads/NeonLogicalAndWorkload.hpp"
#include "workloads/NeonLogicalNotWorkload.hpp"
#include "workloads/NeonLogicalOrWorkload.hpp"
#include "workloads/NeonLstmFloatWorkload.hpp"
#include "workloads/NeonMaximumWorkload.hpp"
#include "workloads/NeonMeanWorkload.hpp"
#include "workloads/NeonMinimumWorkload.hpp"
#include "workloads/NeonMultiplicationWorkload.hpp"
#include "workloads/NeonDivisionWorkload.hpp"
#include "workloads/NeonNegWorkload.hpp"
#include "workloads/NeonNormalizationFloatWorkload.hpp"
#include "workloads/NeonFullyConnectedWorkload.hpp"
#include "workloads/NeonGatherWorkload.hpp"
#include "workloads/NeonGatherNdWorkload.hpp"
#include "workloads/NeonPadWorkload.hpp"
#include "workloads/NeonPermuteWorkload.hpp"
#include "workloads/NeonPooling2dWorkload.hpp"
#include "workloads/NeonPooling3dWorkload.hpp"
#include "workloads/NeonPreluWorkload.hpp"
#include "workloads/NeonQLstmWorkload.hpp"
#include "workloads/NeonQuantizeWorkload.hpp"
#include "workloads/NeonQuantizedLstmWorkload.hpp"
#include "workloads/NeonReduceWorkload.hpp"
#include "workloads/NeonReshapeWorkload.hpp"
#include "workloads/NeonResizeWorkload.hpp"
#include "workloads/NeonRsqrtWorkload.hpp"
#include "workloads/NeonSinWorkload.hpp"
#include "workloads/NeonSliceWorkload.hpp"
#include "workloads/NeonSoftmaxWorkload.hpp"
#include "workloads/NeonSpaceToBatchNdWorkload.hpp"
#include "workloads/NeonSpaceToDepthWorkload.hpp"
#include "workloads/NeonSplitterWorkload.hpp"
#include "workloads/NeonSqrtWorkload.hpp"
#include "workloads/NeonStackWorkload.hpp"
#include "workloads/NeonStridedSliceWorkload.hpp"
#include "workloads/NeonSubtractionWorkload.hpp"
#include "workloads/NeonTransposeConvolution2dWorkload.hpp"
#include "workloads/NeonTransposeWorkload.hpp"
#include "workloads/NeonUnidirectionalSequenceLstmFloatWorkload.hpp"
#include "workloads/NeonUnidirectionalSequenceLstmWorkload.hpp"
#endif

namespace armnn
{

namespace
{

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

template< typename ... Args>
bool IsNeonBackendSupported(Optional<std::string&> reasonIfUnsupported, Args... args)
{
    IgnoreUnused(reasonIfUnsupported, (args)...);
#if defined(ARMCOMPUTENEON_ENABLED)
    return true;
#else
    SetValueChecked(reasonIfUnsupported, "The armnn library has been built without NEON support");
    return false;
#endif
}

template<typename FloatFunc, typename Uint8Func, typename ... Params>
bool IsSupportedForDataTypeNeon(Optional<std::string&> reasonIfUnsupported,
                                DataType dataType,
                                FloatFunc floatFuncPtr,
                                Uint8Func uint8FuncPtr,
                                Params&&... params)
{
    return IsNeonBackendSupported(reasonIfUnsupported) &&
        IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                         dataType,
                                         floatFuncPtr,
                                         floatFuncPtr,
                                         uint8FuncPtr,
                                         &FalseFunc<>,
                                         &FalseFunc<>,
                                         std::forward<Params>(params)...);
}

#if defined(ARMCOMPUTENEON_ENABLED)
template<class FuncType, class... Args>
inline bool IsWorkloadSupported(FuncType& func, Optional<std::string&> reasonIfUnsupported, Args&&... args)
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
    return IsNeonBackendSupported(reasonIfUnsupported, __VA_ARGS__);
#endif
} // anonymous namespace

NeonLayerSupport::NeonLayerSupport(const IBackendInternal::IBackendSpecificModelContextPtr& modelContextPtr)
    : m_ModelContextPtr(modelContextPtr)
{
}

NeonLayerSupport::NeonLayerSupport()
    : m_ModelContextPtr(nullptr)
{
}

bool IsLayerTypeSupported(const LayerType& type,
                          const std::vector<TensorInfo>& infos,
                          const BaseDescriptor& descriptor,
                          const Optional<LstmInputParamsInfo>& lstmParamsInfo,
                          const Optional<QuantizedLstmInputParamsInfo>& quantizedLstmParamsInfo,
                          Optional<std::string&> reasonIfUnsupported,
                          const NeonLayerSupport& support)
{
    switch (type)
    {
        case LayerType::Activation:
            return support.IsActivationSupported(infos[0],
                                                 infos[1],
                                                 *(PolymorphicDowncast<const ActivationDescriptor*>(&descriptor)),
                                                 reasonIfUnsupported);
        case LayerType::Addition:
            return support.IsAdditionSupported(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::ArgMinMax:
            return support.IsArgMinMaxSupported(infos[0],
                                                infos[1],
                                                *(PolymorphicDowncast<const ArgMinMaxDescriptor*>(&descriptor)),
                                                reasonIfUnsupported);
        case LayerType::BatchMatMul:
            return support.IsBatchMatMulSupported(infos[0],
                                                  infos[1],
                                                  infos[2],
                                                  *(PolymorphicDowncast<const BatchMatMulDescriptor*>(&descriptor)),
                                                  reasonIfUnsupported);
        case LayerType::BatchNormalization:
            return support.IsBatchNormalizationSupported(infos[0],
                                                         infos[1],
                                                         infos[2],
                                                         infos[3],
                                                         infos[4],
                                                         infos[5],
                                                         *(PolymorphicDowncast<const
                                                             BatchNormalizationDescriptor*>(&descriptor)),
                                                         reasonIfUnsupported);
        case LayerType::BatchToSpaceNd:
            return support.IsBatchToSpaceNdSupported(infos[0],
                                                     infos[1],
                                                     *(PolymorphicDowncast<const
                                                        BatchToSpaceNdDescriptor*>(&descriptor)),
                                                     reasonIfUnsupported);
        case LayerType::Cast:
            return support.IsCastSupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::ChannelShuffle:
            return support.IsChannelShuffleSupported(infos[0],
                                                     infos[1],
                                                     *(PolymorphicDowncast<const
                                                         ChannelShuffleDescriptor*>(&descriptor)),
                                                     reasonIfUnsupported);
        case LayerType::Comparison:
            return support.IsComparisonSupported(infos[0],
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
            return support.IsConcatSupported(inputInfos,
                                             infos[infos.size() - 1],
                                             *(PolymorphicDowncast<const OriginsDescriptor*>(&descriptor)),
                                             reasonIfUnsupported);
        }
        case LayerType::Constant:
            return support.IsConstantSupported(infos[0], reasonIfUnsupported);
        case LayerType::ConvertFp16ToFp32:
            return support.IsConvertFp16ToFp32Supported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::ConvertFp32ToFp16:
            return support.IsConvertFp32ToFp16Supported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::Convolution2d:
        {
            if (infos.size() != 4)
            {
                throw InvalidArgumentException("Invalid number of TransposeConvolution2d TensorInfos. "
                                               "TensorInfos should be of format: {input, output, weights, biases}.");
            }

            auto desc = *(PolymorphicDowncast<const Convolution2dDescriptor*>(&descriptor));
            if (infos[3] == TensorInfo())
            {
                return support.IsConvolution2dSupported(infos[0],
                                                        infos[1],
                                                        desc,
                                                        infos[2],
                                                        EmptyOptional(),
                                                        reasonIfUnsupported);
            }
            else
            {
                return support.IsConvolution2dSupported(infos[0],
                                                        infos[1],
                                                        desc,
                                                        infos[2],
                                                        infos[3],
                                                        reasonIfUnsupported);
            }
        }
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
                return support.IsConvolution3dSupported(infos[0],
                                                        infos[1],
                                                        desc,
                                                        infos[2],
                                                        EmptyOptional(),
                                                        reasonIfUnsupported);
            }
            else
            {
                return support.IsConvolution3dSupported(infos[0],
                                                        infos[1],
                                                        desc,
                                                        infos[2],
                                                        infos[3],
                                                        reasonIfUnsupported);
            }
        }
        case LayerType::DepthToSpace:
            return support.IsDepthToSpaceSupported(infos[0],
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
                return support.IsDepthwiseConvolutionSupported(infos[0],
                                                               infos[1],
                                                               desc,
                                                               infos[2],
                                                               EmptyOptional(),
                                                               reasonIfUnsupported);
            }
            else
            {
                return support.IsDepthwiseConvolutionSupported(infos[0],
                                                               infos[1],
                                                               desc,
                                                               infos[2],
                                                               infos[3],
                                                               reasonIfUnsupported);
            }
        }
        case LayerType::Dequantize:
            return support.IsDequantizeSupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::DetectionPostProcess:
        {
            auto desc = *(PolymorphicDowncast<const DetectionPostProcessDescriptor*>(&descriptor));
            return support.IsDetectionPostProcessSupported(infos[0],
                                                           infos[1],
                                                           infos[2],
                                                           infos[3],
                                                           infos[4],
                                                           infos[5],
                                                           infos[6],
                                                           desc,
                                                           reasonIfUnsupported);
        }
        case LayerType::Division:
            return support.IsDivisionSupported(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::ElementwiseBinary:
        {
            auto desc = *(PolymorphicDowncast<const ElementwiseBinaryDescriptor *>(&descriptor));

            switch (desc.m_Operation)
            {
                case BinaryOperation::Add:
                    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonAdditionWorkloadValidate,
                                                   reasonIfUnsupported,
                                                   infos[0],
                                                   infos[1],
                                                   infos[2],
                                                   nullptr);
                case BinaryOperation::Div:
                    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonDivisionWorkloadValidate,
                                                   reasonIfUnsupported,
                                                   infos[0],
                                                   infos[1],
                                                   infos[2],
                                                   nullptr);
                case BinaryOperation::Maximum:
                    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonMaximumWorkloadValidate,
                                                   reasonIfUnsupported,
                                                   infos[0],
                                                   infos[1],
                                                   infos[2]);
                case BinaryOperation::Minimum:
                    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonMinimumWorkloadValidate,
                                                   reasonIfUnsupported,
                                                   infos[0],
                                                   infos[1],
                                                   infos[2]);
                case BinaryOperation::Mul:
                    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonMultiplicationWorkloadValidate,
                                                   reasonIfUnsupported,
                                                   infos[0],
                                                   infos[1],
                                                   infos[2],
                                                   nullptr);
                case BinaryOperation::Sub:
                    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonSubtractionWorkloadValidate,
                                                   reasonIfUnsupported,
                                                   infos[0],
                                                   infos[1],
                                                   infos[2],
                                                   nullptr);
                default:
                    return false;
            }
        }
        case LayerType::ElementwiseUnary:
            return support.IsElementwiseUnarySupported(infos[0],
                                                       infos[1],
                                                       *(PolymorphicDowncast<const
                                                           ElementwiseUnaryDescriptor*>(&descriptor)),
                                                       reasonIfUnsupported);
        case LayerType::Fill:
            return support.IsFillSupported(infos[0],
                                           infos[1],
                                           *(PolymorphicDowncast<const FillDescriptor*>(&descriptor)),
                                           reasonIfUnsupported);
        case LayerType::Floor:
            return support.IsFloorSupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::FullyConnected:
            return support.IsFullyConnectedSupported(infos[0],
                                                     infos[1],
                                                     infos[2],
                                                     infos[3],
                                                     *(PolymorphicDowncast<const
                                                         FullyConnectedDescriptor*>(&descriptor)),
                                                     reasonIfUnsupported);
        case LayerType::Gather:
            return support.IsGatherSupported(infos[0],
                                             infos[1],
                                             infos[2],
                                             *(PolymorphicDowncast<const GatherDescriptor*>(&descriptor)),
                                             reasonIfUnsupported);
        case LayerType::GatherNd:
            return support.IsGatherNdSupported(infos[0],
                                               infos[1],
                                               infos[2],
                                               reasonIfUnsupported);
        case LayerType::Input:
            return support.IsInputSupported(infos[0], reasonIfUnsupported);
        case LayerType::InstanceNormalization:
            return support.IsInstanceNormalizationSupported(infos[0],
                                                            infos[1],
                                                            *(PolymorphicDowncast<const
                                                                InstanceNormalizationDescriptor*>(&descriptor)),
                                                            reasonIfUnsupported);
        case LayerType::L2Normalization:
            return support.IsL2NormalizationSupported(infos[0],
                                                      infos[1],
                                                      *(PolymorphicDowncast<const
                                                          L2NormalizationDescriptor*>(&descriptor)),
                                                      reasonIfUnsupported);
        case LayerType::LogicalBinary:
            return support.IsLogicalBinarySupported(infos[0],
                                                    infos[1],
                                                    infos[2],
                                                    *(PolymorphicDowncast<const
                                                        LogicalBinaryDescriptor*>(&descriptor)),
                                                    reasonIfUnsupported);
        case LayerType::LogSoftmax:
            return support.IsLogSoftmaxSupported(infos[0],
                                                 infos[1],
                                                 *(PolymorphicDowncast<const LogSoftmaxDescriptor*>(&descriptor)),
                                                 reasonIfUnsupported);
        case LayerType::Lstm:
            return support.IsLstmSupported(infos[0],
                                           infos[1],
                                           infos[2],
                                           infos[3],
                                           infos[4],
                                           infos[5],
                                           infos[6],
                                           *(PolymorphicDowncast<const LstmDescriptor*>(&descriptor)),
                                           lstmParamsInfo.value(),
                                           reasonIfUnsupported);
        case LayerType::Map:
            return true;
        case LayerType::Maximum:
            return support.IsMaximumSupported(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::Mean:
            return support.IsMeanSupported(infos[0],
                                           infos[1],
                                           *(PolymorphicDowncast<const MeanDescriptor*>(&descriptor)),
                                           reasonIfUnsupported);
        case LayerType::MemCopy:
            return support.IsMemCopySupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::MemImport:
            return support.IsMemImportSupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::Merge:
            return support.IsMergeSupported(infos[0],
                                                      infos[1],
                                                      infos[2],
                                                      reasonIfUnsupported);
        case LayerType::Minimum:
            return support.IsMinimumSupported(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::Multiplication:
            return support.IsMultiplicationSupported(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::Normalization:
            return support.IsNormalizationSupported(infos[0],
                                                    infos[1],
                                                    *(PolymorphicDowncast<const
                                                        NormalizationDescriptor*>(&descriptor)),
                                                    reasonIfUnsupported);
        case LayerType::Output:
            return support.IsOutputSupported(infos[0], reasonIfUnsupported);
        case LayerType::Pad:
            return support.IsPadSupported(infos[0],
                                          infos[1],
                                          *(PolymorphicDowncast<const PadDescriptor*>(&descriptor)),
                                          reasonIfUnsupported);
        case LayerType::Permute:
            return support.IsPermuteSupported(infos[0],
                                              infos[1],
                                              *(PolymorphicDowncast<const PermuteDescriptor*>(&descriptor)),
                                              reasonIfUnsupported);
        case LayerType::Pooling2d:
            return support.IsPooling2dSupported(infos[0],
                                                infos[1],
                                                *(PolymorphicDowncast<const Pooling2dDescriptor*>(&descriptor)),
                                                reasonIfUnsupported);
        case LayerType::Pooling3d:
            return support.IsPooling3dSupported(infos[0],
                                                infos[1],
                                                *(PolymorphicDowncast<const Pooling3dDescriptor*>(&descriptor)),
                                                reasonIfUnsupported);
        case LayerType::Prelu:
            return support.IsPreluSupported(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::QLstm:
            return support.IsQLstmSupported(infos[0],
                                            infos[1],
                                            infos[2],
                                            infos[3],
                                            infos[4],
                                            infos[5],
                                            *(PolymorphicDowncast<const QLstmDescriptor*>(&descriptor)),
                                            lstmParamsInfo.value(),
                                            reasonIfUnsupported);
        case LayerType::Quantize:
            return support.IsQuantizeSupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::QuantizedLstm:
            return support.IsQuantizedLstmSupported(infos[0],
                                                    infos[1],
                                                    infos[2],
                                                    infos[3],
                                                    infos[4],
                                                    quantizedLstmParamsInfo.value(),
                                                    reasonIfUnsupported);
        case LayerType::Rank:
            return true;
        case LayerType::Reshape:
            return support.IsReshapeSupported(infos[0],
                                              infos[1],
                                              *(PolymorphicDowncast<const ReshapeDescriptor*>(&descriptor)),
                                              reasonIfUnsupported);
        case LayerType::Resize:
            return support.IsResizeSupported(infos[0],
                                             infos[1],
                                             *(PolymorphicDowncast<const ResizeDescriptor*>(&descriptor)),
                                             reasonIfUnsupported);
        case LayerType::Reduce:
            return support.IsReduceSupported(infos[0],
                                             infos[1],
                                             *(PolymorphicDowncast<const ReduceDescriptor*>(&descriptor)),
                                             reasonIfUnsupported);
        case LayerType::Shape:
            return support.IsShapeSupported(infos[0],
                                            infos[1],
                                            reasonIfUnsupported);
        case LayerType::Slice:
            return support.IsSliceSupported(infos[0],
                                            infos[1],
                                            *(PolymorphicDowncast<const SliceDescriptor*>(&descriptor)),
                                            reasonIfUnsupported);
        case LayerType::Softmax:
            return support.IsSoftmaxSupported(infos[0],
                                              infos[1],
                                              *(PolymorphicDowncast<const SoftmaxDescriptor*>(&descriptor)),
                                              reasonIfUnsupported);
        case LayerType::SpaceToBatchNd:
            return support.IsSpaceToBatchNdSupported(infos[0],
                                                     infos[1],
                                                     *(PolymorphicDowncast<const
                                                        SpaceToBatchNdDescriptor*>(&descriptor)),
                                                     reasonIfUnsupported);
        case LayerType::SpaceToDepth:
            return support.IsSpaceToDepthSupported(infos[0],
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
            return support.IsSplitterSupported(infos[0],
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
            return support.IsStackSupported(inputInfos,
                                            infos[infos.size() - 1],
                                            *(PolymorphicDowncast<const StackDescriptor*>(&descriptor)),
                                            reasonIfUnsupported);
        }
        case LayerType::StridedSlice:
            return support.IsStridedSliceSupported(infos[0],
                                                   infos[1],
                                                   *(PolymorphicDowncast<const StridedSliceDescriptor*>(&descriptor)),
                                                   reasonIfUnsupported);
        case LayerType::Subtraction:
            return support.IsSubtractionSupported(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::Transpose:
            return support.IsTransposeSupported(infos[0],
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
                return support.IsTransposeConvolution2dSupported(infos[0],
                                                                 infos[1],
                                                                 desc,
                                                                 infos[2],
                                                                 EmptyOptional(),
                                                                 reasonIfUnsupported);
            }
            else
            {
                return support.IsTransposeConvolution2dSupported(infos[0],
                                                                 infos[1],
                                                                 desc,
                                                                 infos[2],
                                                                 infos[3],
                                                                 reasonIfUnsupported);
            }
        }
        case LayerType::UnidirectionalSequenceLstm:
        {
            auto desc = *(PolymorphicDowncast<const UnidirectionalSequenceLstmDescriptor*>(&descriptor));
            return support.IsUnidirectionalSequenceLstmSupported(infos[0],
                                                                 infos[1],
                                                                 infos[2],
                                                                 infos[3],
                                                                 infos[4],
                                                                 infos[5],
                                                                 desc,
                                                                 lstmParamsInfo.value(),
                                                                 reasonIfUnsupported);
        }
        case LayerType::Unmap:
            return true;
        default:
            // layers not supported in neon by default:
            // debug, fakequantization, precompiled,
            // standin, switch
            return false;
    }
}

bool NeonLayerSupport::IsLayerSupported(const LayerType& type,
                                        const std::vector<TensorInfo>& infos,
                                        const BaseDescriptor& descriptor,
                                        const Optional<LstmInputParamsInfo>& lstmParamsInfo,
                                        const Optional<QuantizedLstmInputParamsInfo>& quantizedLstmParamsInfo,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    bool isSupported = IsLayerTypeSupported(type,
                                            infos,
                                            descriptor,
                                            lstmParamsInfo,
                                            quantizedLstmParamsInfo,
                                            reasonIfUnsupported,
                                            *this);

    // For android-nn-driver and support library, to run FP16 operations on CpuAcc we need at least v8.2
    // architecture. If the available architecture is older than v8.2, we can check if the operator is
    // supported by changing operator inputs & outputs to be FP32.
    // This does not change the operator datatype in the above parsers to be FP32. We are simply reporting
    // to the parsers if the operator can supported in ArmNN. We will then re-enter ArmNN (Network.cpp)
    // where we will recheck IsLayerSupported() on the FP16 datatype, update the operator to be FP32,
    // and, insert convert layers around the FP32 operator.
    if (reasonIfUnsupported.has_value())
    {
        std::string checkStr = "This CPU architecture does not support F16 data type, you need v8.2 or above";
        if (!isSupported
            && reasonIfUnsupported.value().find(checkStr) != std::string::npos)
        {
            std::vector<TensorInfo> newInfos;
            for (auto               info: infos)
            {
                newInfos.emplace_back(OverrideDataType(info, DataType::Float32));
            }

            std::string tmpString;
            return IsLayerTypeSupported(type,
                                        newInfos,
                                        descriptor,
                                        lstmParamsInfo,
                                        quantizedLstmParamsInfo,
                                        tmpString,
                                        *this);
        }
    }

    return isSupported;
}

bool NeonLayerSupport::IsActivationSupported(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const ActivationDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonActivationWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsAdditionSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonAdditionWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output,
                                   nullptr);
}

bool NeonLayerSupport::IsArgMinMaxSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const ArgMinMaxDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonArgMinMaxWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsBatchMatMulSupported(const TensorInfo& inputX,
                                              const TensorInfo& inputY,
                                              const TensorInfo& output,
                                              const BatchMatMulDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonBatchMatMulValidate,
                                   reasonIfUnsupported,
                                   inputX,
                                   inputY,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsBatchNormalizationSupported(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const TensorInfo& mean,
                                                     const TensorInfo& var,
                                                     const TensorInfo& beta,
                                                     const TensorInfo& gamma,
                                                     const BatchNormalizationDescriptor& descriptor,
                                                     Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonBatchNormalizationValidate,
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

bool NeonLayerSupport::IsBatchToSpaceNdSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const BatchToSpaceNdDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonBatchToSpaceNdWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsCastSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonCastValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output);
}

bool NeonLayerSupport::IsChannelShuffleSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const ChannelShuffleDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonChannelShuffleValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsComparisonSupported(const TensorInfo& input0,
                                             const TensorInfo& input1,
                                             const TensorInfo& output,
                                             const ComparisonDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported) const
{

    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonComparisonWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsConcatSupported(const std::vector<const TensorInfo*> inputs,
                                         const TensorInfo& output,
                                         const OriginsDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    if (descriptor.GetNumDimensions() <= descriptor.GetConcatAxis())
    {
        SetValueChecked(reasonIfUnsupported, "Neon Concat: Concat axis > Number of dimensions.");
        return false;
    }

    unsigned int concatInnerAxis = (descriptor.GetNumDimensions() - descriptor.GetConcatAxis()) - 1;
    if(concatInnerAxis < 3) // Width, height, or channels
    {
        FORWARD_WORKLOAD_VALIDATE_FUNC(NeonConcatWorkloadValidate,
                                       reasonIfUnsupported,
                                       inputs,
                                       output,
                                       descriptor);
    }
    else if (concatInnerAxis == 3)
    {
        for (auto& input : inputs)
        {
            if (input && !output.IsTypeSpaceMatch(*input)) // Cannot use sub-tensors if the types are not same space
            {
                SetValueChecked(reasonIfUnsupported, "Neon Concat: Types and quantization parameters must match.");
                return false;
            }
        }
        return true; // Sub-tensors support concat along batch
    }
    else // > 4 dimensions not supported.
    {
        SetValueChecked(reasonIfUnsupported, "Neon Concat: Maximum of 4 dimensions supported.");
        return false;
    }
}

bool NeonLayerSupport::IsConstantSupported(const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonConstantWorkloadValidate,
                                   reasonIfUnsupported,
                                   output);
}

bool NeonLayerSupport::IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    armnn::IgnoreUnused(input);
    armnn::IgnoreUnused(output);
    armnn::IgnoreUnused(reasonIfUnsupported);
    return true;
}

bool NeonLayerSupport::IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    armnn::IgnoreUnused(input);
    armnn::IgnoreUnused(output);
    armnn::IgnoreUnused(reasonIfUnsupported);
    return true;
}

bool NeonLayerSupport::IsConvolution2dSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const Convolution2dDescriptor& descriptor,
                                                const TensorInfo& weights,
                                                const Optional<TensorInfo>& biases,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    bool isFastMathEnabled = false;
#if defined(ARMCOMPUTENEON_ENABLED)
    if (m_ModelContextPtr)
    {
        if (m_ModelContextPtr.get() != nullptr)
        {
            auto modelOptions = dynamic_cast<NeonBackendModelContext*>(m_ModelContextPtr.get());
            if (modelOptions)
            {
                isFastMathEnabled = modelOptions->IsFastMathEnabled();
            }
        }
    }
#endif

    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonConvolution2dWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases,
                                   isFastMathEnabled,
                                   nullptr);
}

bool NeonLayerSupport::IsConvolution3dSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const Convolution3dDescriptor& descriptor,
                                                const TensorInfo& weights,
                                                const Optional<TensorInfo>& biases,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    bool isFastMathEnabled = false;
#if defined(ARMCOMPUTENEON_ENABLED)
    if (m_ModelContextPtr)
    {
        if (m_ModelContextPtr.get() != nullptr)
        {
            auto modelOptions = dynamic_cast<NeonBackendModelContext*>(m_ModelContextPtr.get());
            if (modelOptions)
            {
                isFastMathEnabled = modelOptions->IsFastMathEnabled();
            }
        }
    }
#endif

    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonConvolution3dWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases,
                                   isFastMathEnabled,
                                   nullptr);
}

bool NeonLayerSupport::IsDepthToSpaceSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const DepthToSpaceDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonDepthToSpaceWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const DepthwiseConvolution2dDescriptor& descriptor,
                                                       const TensorInfo& weights,
                                                       const Optional<TensorInfo>& biases,
                                                       Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonDepthwiseConvolutionWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases,
                                   nullptr);
}

bool NeonLayerSupport::IsDequantizeSupported(const TensorInfo& input,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonDequantizeWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output);
}

bool NeonLayerSupport::IsDilatedDepthwiseConvolutionSupported(const TensorInfo& input,
                                                              const TensorInfo& output,
                                                              const DepthwiseConvolution2dDescriptor& descriptor,
                                                              const TensorInfo& weights,
                                                              const Optional<TensorInfo>& biases,
                                                              Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonDepthwiseConvolutionWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases,
                                   nullptr);
}

bool NeonLayerSupport::IsElementwiseUnarySupported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const ElementwiseUnaryDescriptor& descriptor,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    switch(descriptor.m_Operation)
    {
        case UnaryOperation::Abs:
            FORWARD_WORKLOAD_VALIDATE_FUNC(NeonAbsWorkloadValidate,
                                           reasonIfUnsupported,
                                           input,
                                           output);
        case UnaryOperation::Exp:
            FORWARD_WORKLOAD_VALIDATE_FUNC(NeonExpWorkloadValidate,
                                           reasonIfUnsupported,
                                           input,
                                           output);
        case UnaryOperation::LogicalNot:
            FORWARD_WORKLOAD_VALIDATE_FUNC(NeonLogicalNotWorkloadValidate,
                                           reasonIfUnsupported,
                                           input,
                                           output);
       case UnaryOperation::Log:
            FORWARD_WORKLOAD_VALIDATE_FUNC(NeonLogWorkloadValidate,
                                           reasonIfUnsupported,
                                           input,
                                           output);
        case UnaryOperation::Neg:
            FORWARD_WORKLOAD_VALIDATE_FUNC(NeonNegWorkloadValidate,
                                           reasonIfUnsupported,
                                           input,
                                           output);
        case UnaryOperation::Rsqrt:
            FORWARD_WORKLOAD_VALIDATE_FUNC(NeonRsqrtWorkloadValidate,
                                           reasonIfUnsupported,
                                           input,
                                           output);
        case UnaryOperation::Sin:
            FORWARD_WORKLOAD_VALIDATE_FUNC(NeonSinWorkloadValidate,
                                           reasonIfUnsupported,
                                           input,
                                           output);
        case UnaryOperation::Sqrt:
            FORWARD_WORKLOAD_VALIDATE_FUNC(NeonSqrtWorkloadValidate,
                                           reasonIfUnsupported,
                                           input,
                                           output);
        default:
            return false;
    }
}

bool NeonLayerSupport::IsFillSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const FillDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    armnn::IgnoreUnused(input);
    armnn::IgnoreUnused(output);
    armnn::IgnoreUnused(descriptor);

    return IsNeonBackendSupported(reasonIfUnsupported);
}

bool NeonLayerSupport::IsFloorSupported(const TensorInfo& input,
                                        const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    armnn::IgnoreUnused(output);
    return IsNeonBackendSupported(reasonIfUnsupported) &&
           IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                         input.GetDataType(),
                                         &FalseFuncF16<>,
                                         &TrueFunc<>,
                                         &FalseFuncU8<>,
                                         &FalseFuncI32<>,
                                         &FalseFuncU8<>);
}

bool NeonLayerSupport::IsFullyConnectedSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const TensorInfo& weights,
                                                 const TensorInfo& biases,
                                                 const FullyConnectedDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonFullyConnectedWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   weights,
                                   biases,
                                   descriptor,
                                   nullptr);
}

bool NeonLayerSupport::IsGatherSupported(const TensorInfo& input0,
                                         const TensorInfo& input1,
                                         const TensorInfo& output,
                                         const GatherDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonGatherWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsGatherNdSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonGatherNdWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output);
}

bool NeonLayerSupport::IsInputSupported(const TensorInfo& input,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return IsNeonBackendSupported(reasonIfUnsupported, input);
}

bool NeonLayerSupport::IsInstanceNormalizationSupported(const TensorInfo& input,
                                                        const TensorInfo& output,
                                                        const InstanceNormalizationDescriptor& descriptor,
                                                        Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonInstanceNormalizationWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsL2NormalizationSupported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const L2NormalizationDescriptor& descriptor,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonL2NormalizationWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool NeonLayerSupport::IsLogicalBinarySupported(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output,
                                                const LogicalBinaryDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    switch(descriptor.m_Operation)
    {
        case LogicalBinaryOperation::LogicalAnd:
            FORWARD_WORKLOAD_VALIDATE_FUNC(NeonLogicalAndWorkloadValidate,
                                           reasonIfUnsupported,
                                           input0,
                                           input1,
                                           output);
        case LogicalBinaryOperation::LogicalOr:
            FORWARD_WORKLOAD_VALIDATE_FUNC(NeonLogicalOrWorkloadValidate,
                                           reasonIfUnsupported,
                                           input0,
                                           input1,
                                           output);
        default:
            return false;
    }
}

bool NeonLayerSupport::IsLogSoftmaxSupported(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const LogSoftmaxDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonLogSoftmaxWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool NeonLayerSupport::IsLstmSupported(const TensorInfo& input,
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
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonLstmFloatWorkloadValidate,
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

bool NeonLayerSupport::IsMaximumSupported(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonMaximumWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output);
}

bool NeonLayerSupport::IsMeanSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const MeanDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonMeanWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsMinimumSupported(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonMinimumWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output);
}

bool NeonLayerSupport::IsMultiplicationSupported(const TensorInfo& input0,
                                                 const TensorInfo& input1,
                                                 const TensorInfo& output,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonMultiplicationWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output,
                                   nullptr);
}

bool NeonLayerSupport::IsDivisionSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonDivisionWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output,
                                   nullptr);
}

bool NeonLayerSupport::IsNormalizationSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const NormalizationDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonNormalizationWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsOutputSupported(const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return IsNeonBackendSupported(reasonIfUnsupported, output);
}

bool NeonLayerSupport::IsPadSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const PadDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonPadWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsPermuteSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const PermuteDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonPermuteWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool NeonLayerSupport::IsPooling2dSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const Pooling2dDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonPooling2dWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool NeonLayerSupport::IsPooling3dSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const Pooling3dDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonPooling3dWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool NeonLayerSupport::IsPreluSupported(const armnn::TensorInfo &input,
                                        const armnn::TensorInfo &alpha,
                                        const armnn::TensorInfo &output,
                                        armnn::Optional<std::string &> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonPreluWorkloadValidate, reasonIfUnsupported, input, alpha, output);
}

bool NeonLayerSupport::IsQLstmSupported(const TensorInfo& input,
                                        const TensorInfo& previousOutputIn,
                                        const TensorInfo& previousCellStateIn,
                                        const TensorInfo& outputStateOut,
                                        const TensorInfo& cellStateOut,
                                        const TensorInfo& output,
                                        const QLstmDescriptor& descriptor,
                                        const LstmInputParamsInfo& paramsInfo,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    // Check required here in order to pass IsLayerSupported for datatypes tests
    if (input.GetDataType()               == armnn::DataType::QAsymmS8 &&
        previousOutputIn.GetDataType()    == armnn::DataType::QAsymmS8 &&
        previousCellStateIn.GetDataType() == armnn::DataType::QSymmS16 &&
        outputStateOut.GetDataType()      == armnn::DataType::QAsymmS8 &&
        cellStateOut.GetDataType()        == armnn::DataType::QSymmS16 &&
        output.GetDataType()              == armnn::DataType::QAsymmS8)
    {
        FORWARD_WORKLOAD_VALIDATE_FUNC(NeonQLstmWorkloadValidate,
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

bool NeonLayerSupport::IsQuantizeSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonQuantizeWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output);
}

bool NeonLayerSupport::IsQuantizedLstmSupported(const TensorInfo& input,
                                                const TensorInfo& cellStateIn,
                                                const TensorInfo& outputStateIn,
                                                const TensorInfo& cellStateOut,
                                                const TensorInfo& outputStateOut,
                                                const QuantizedLstmInputParamsInfo& paramsInfo,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonQuantizedLstmWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   cellStateIn,
                                   outputStateIn,
                                   cellStateOut,
                                   outputStateOut,
                                   paramsInfo);
}

bool NeonLayerSupport::IsReduceSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const ReduceDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonReduceWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsReshapeSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const ReshapeDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    armnn::IgnoreUnused(descriptor);
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonReshapeWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output);
}

bool NeonLayerSupport::IsResizeSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const ResizeDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonResizeWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsSliceSupported(const TensorInfo& input,
                                        const TensorInfo& output,
                                        const SliceDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonSliceWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsSoftmaxSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const SoftmaxDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonSoftmaxWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool NeonLayerSupport::IsSpaceToBatchNdSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const SpaceToBatchNdDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonSpaceToBatchNdWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsSpaceToDepthSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const SpaceToDepthDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonSpaceToDepthWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsSplitterSupported(const TensorInfo& input,
                                           const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                                           const ViewsDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported) const
{
#if defined(ARMCOMPUTENEON_ENABLED)
    // Split along the last dimension, cannot use sub-tensors
    // as width and height of the sub-tensors do not match
    // the width and height of the parent tensor
    // in case of input with more than 2D.
    std::set<unsigned int> splitAxis = ComputeSplitAxis(descriptor, input.GetShape());
    if (descriptor.GetNumDimensions() > 2 && splitAxis.size() == 1 &&
        *splitAxis.begin() == descriptor.GetNumDimensions() - 1 )
    {
        FORWARD_WORKLOAD_VALIDATE_FUNC(NeonSplitterWorkloadValidate,
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
            SetValueChecked(reasonIfUnsupported, "Neon Splitter: Types and quantization parameters must match.");
            return false;
        }
    }
    return true;
}

bool NeonLayerSupport::IsStackSupported(const std::vector<const TensorInfo*>& inputs,
                                        const TensorInfo& output,
                                        const StackDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonStackWorkloadValidate,
                                   reasonIfUnsupported,
                                   inputs,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsStridedSliceSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const StridedSliceDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonStridedSliceWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsSubtractionSupported(const TensorInfo& input0,
                                              const TensorInfo& input1,
                                              const TensorInfo& output,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonSubtractionWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output,
                                   nullptr);
}

bool NeonLayerSupport::IsTransposeConvolution2dSupported(const TensorInfo& input,
                                                         const TensorInfo& output,
                                                         const TransposeConvolution2dDescriptor& descriptor,
                                                         const TensorInfo& weights,
                                                         const Optional<TensorInfo>& biases,
                                                         Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonTransposeConvolution2dWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases);
}

bool NeonLayerSupport::IsTransposeSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const TransposeDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonTransposeWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool NeonLayerSupport::IsUnidirectionalSequenceLstmSupported(const TensorInfo& input,
                                                             const TensorInfo& outputStateIn,
                                                             const TensorInfo& cellStateIn,
                                                             const TensorInfo& outputStateOut,
                                                             const TensorInfo& cellStateOut,
                                                             const TensorInfo& output,
                                                             const UnidirectionalSequenceLstmDescriptor& descriptor,
                                                             const LstmInputParamsInfo& paramsInfo,
                                                             Optional<std::string&> reasonIfUnsupported) const
{
    if (input.GetDataType() == armnn::DataType::QAsymmS8 &&
        outputStateIn.GetDataType() == armnn::DataType::QAsymmS8 &&
        cellStateIn.GetDataType() == armnn::DataType::QSymmS16 &&
        outputStateOut.GetDataType() == armnn::DataType::QAsymmS8 &&
        cellStateOut.GetDataType() == armnn::DataType::QSymmS16 &&
        output.GetDataType() == armnn::DataType::QAsymmS8)
    {
        FORWARD_WORKLOAD_VALIDATE_FUNC(NeonUnidirectionalSequenceLstmWorkloadValidate,
                                       reasonIfUnsupported,
                                       input,
                                       outputStateIn,
                                       cellStateIn,
                                       outputStateOut,
                                       cellStateOut,
                                       output,
                                       descriptor,
                                       paramsInfo);
    }
    else
    {
        FORWARD_WORKLOAD_VALIDATE_FUNC(NeonUnidirectionalSequenceLstmFloatWorkloadValidate,
                                       reasonIfUnsupported,
                                       input,
                                       outputStateIn,
                                       cellStateIn,
                                       outputStateOut,
                                       cellStateOut,
                                       output,
                                       descriptor,
                                       paramsInfo);
    }
}

} // namespace armnn
