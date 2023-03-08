//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <Layer.hpp>
#include <armnn/backends/MemCopyWorkload.hpp>
#include <backendsCommon/MemImportWorkload.hpp>
#include <backendsCommon/MakeWorkloadHelper.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include "RefWorkloadFactory.hpp"
#include "RefBackendId.hpp"
#include "workloads/RefWorkloads.hpp"
#include "RefTensorHandle.hpp"


namespace armnn
{

namespace
{
static const BackendId s_Id{RefBackendId()};
}
template <typename F32Workload, typename U8Workload, typename QueueDescriptorType>
std::unique_ptr<IWorkload> RefWorkloadFactory::MakeWorkload(const QueueDescriptorType& descriptor,
                                                            const WorkloadInfo& info) const
{
    return MakeWorkloadHelper<NullWorkload, F32Workload, U8Workload, NullWorkload, NullWorkload, NullWorkload>
           (descriptor, info);
}

template <DataType ArmnnType>
bool IsDataType(const WorkloadInfo& info)
{
    auto checkType = [](const TensorInfo& tensorInfo) {return tensorInfo.GetDataType() == ArmnnType;};
    auto it = std::find_if(std::begin(info.m_InputTensorInfos), std::end(info.m_InputTensorInfos), checkType);
    if (it != std::end(info.m_InputTensorInfos))
    {
        return true;
    }
    it = std::find_if(std::begin(info.m_OutputTensorInfos), std::end(info.m_OutputTensorInfos), checkType);
    if (it != std::end(info.m_OutputTensorInfos))
    {
        return true;
    }
    return false;
}

bool IsSigned32(const WorkloadInfo& info)
{
    return IsDataType<DataType::Signed32>(info);
}

bool IsBFloat16(const WorkloadInfo& info)
{
    return IsDataType<DataType::BFloat16>(info);
}

bool IsFloat16(const WorkloadInfo& info)
{
    return IsDataType<DataType::Float16>(info);
}

bool IsQSymmS16(const WorkloadInfo& info)
{
    return IsDataType<DataType::QSymmS16>(info);
}

bool IsQSymmS8(const WorkloadInfo& info)
{
    return IsDataType<DataType::QSymmS8>(info);
}

bool IsQAsymmS8(const WorkloadInfo& info)
{
    return IsDataType<DataType::QAsymmS8>(info);
}

bool IsQAsymmU8(const WorkloadInfo& info)
{
    return IsDataType<DataType::QAsymmU8>(info);
}

RefWorkloadFactory::RefWorkloadFactory(const std::shared_ptr<RefMemoryManager>& memoryManager)
    : m_MemoryManager(memoryManager)
{
}

RefWorkloadFactory::RefWorkloadFactory()
    : m_MemoryManager(new RefMemoryManager())
{
}

const BackendId& RefWorkloadFactory::GetBackendId() const
{
    return s_Id;
}

bool RefWorkloadFactory::IsLayerSupported(const Layer& layer,
                                          Optional<DataType> dataType,
                                          std::string& outReasonIfUnsupported)
{
    return IWorkloadFactory::IsLayerSupported(s_Id, layer, dataType, outReasonIfUnsupported);
}

bool RefWorkloadFactory::IsLayerSupported(const IConnectableLayer& layer,
                                          Optional<DataType> dataType,
                                          std::string& outReasonIfUnsupported,
                                          const ModelOptions& modelOptions)
{
    return IWorkloadFactory::IsLayerSupported(s_Id, layer, dataType, outReasonIfUnsupported, modelOptions);
}

std::unique_ptr<ITensorHandle> RefWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                      const bool isMemoryManaged) const
{
    if (isMemoryManaged)
    {
        return std::make_unique<RefTensorHandle>(tensorInfo, m_MemoryManager);
    }
    else
    {
        return std::make_unique<RefTensorHandle>(tensorInfo);
    }
}

std::unique_ptr<ITensorHandle> RefWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                      DataLayout dataLayout,
                                                                      const bool isMemoryManaged) const
{
    // For Ref it is okay to make the TensorHandle memory managed as it can also store a pointer
    // to unmanaged memory. This also ensures memory alignment.
    IgnoreUnused(isMemoryManaged, dataLayout);

    if (isMemoryManaged)
    {
        return std::make_unique<RefTensorHandle>(tensorInfo, m_MemoryManager);
    }
    else
    {
        return std::make_unique<RefTensorHandle>(tensorInfo);
    }
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateWorkload(LayerType type,
                                                              const QueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    switch(type)
    {
        case LayerType::Activation :
        {
            auto activationQueueDescriptor = PolymorphicDowncast<const ActivationQueueDescriptor*>(&descriptor);
            return std::make_unique<RefActivationWorkload>(*activationQueueDescriptor, info);
        }
        case LayerType::Addition :
        {
            auto additionQueueDescriptor = PolymorphicDowncast<const AdditionQueueDescriptor*>(&descriptor);

            if (info.m_InputTensorInfos[0].GetDataType() == armnn::DataType::Signed32)
            {
                return std::make_unique<RefAdditionWorkload<int32_t>>(*additionQueueDescriptor, info);
            }
            else
            {
                return std::make_unique<RefAdditionWorkload<float>>(*additionQueueDescriptor, info);
            }
        }
        case LayerType::ArgMinMax :
        {
            auto argMinMaxQueueDescriptor = PolymorphicDowncast<const ArgMinMaxQueueDescriptor*>(&descriptor);
            return std::make_unique<RefArgMinMaxWorkload>(*argMinMaxQueueDescriptor, info);
        }
        case LayerType::BatchMatMul:
        {
            auto batchMatMulQueueDescriptor = PolymorphicDowncast<const BatchMatMulQueueDescriptor*>(&descriptor);
            return std::make_unique<RefBatchMatMulWorkload>(*batchMatMulQueueDescriptor, info);
        }
        case LayerType::BatchNormalization :
        {
            auto batchNormQueueDescriptor = PolymorphicDowncast<const BatchNormalizationQueueDescriptor*>(&descriptor);
            return std::make_unique<RefBatchNormalizationWorkload>(*batchNormQueueDescriptor, info);
        }
        case LayerType::BatchToSpaceNd :
        {
            auto batchToSpaceNdQueueDescriptor
                    = PolymorphicDowncast<const BatchToSpaceNdQueueDescriptor*>(&descriptor);
            return std::make_unique<RefBatchToSpaceNdWorkload>(*batchToSpaceNdQueueDescriptor, info);
       }
        case LayerType::Cast :
        {
            auto castQueueDescriptor = PolymorphicDowncast<const CastQueueDescriptor*>(&descriptor);
            return std::make_unique<RefCastWorkload>(*castQueueDescriptor, info);
        }
        case LayerType::ChannelShuffle :
        {
            auto channelShuffleQueueDescriptor
                    = PolymorphicDowncast<const ChannelShuffleQueueDescriptor*>(&descriptor);
            return std::make_unique<RefChannelShuffleWorkload>(*channelShuffleQueueDescriptor, info);
        }
        case LayerType::Comparison :
        {
            auto comparisonQueueDescriptor = PolymorphicDowncast<const ComparisonQueueDescriptor*>(&descriptor);
            return std::make_unique<RefComparisonWorkload>(*comparisonQueueDescriptor, info);
        }
        case LayerType::Concat :
        {
            auto concatQueueDescriptor = PolymorphicDowncast<const ConcatQueueDescriptor*>(&descriptor);
            return std::make_unique<RefConcatWorkload>(*concatQueueDescriptor, info);
        }
        case LayerType::Constant :
        {
            auto constantQueueDescriptor = PolymorphicDowncast<const ConstantQueueDescriptor*>(&descriptor);
            return std::make_unique<RefConstantWorkload>(*constantQueueDescriptor, info);
        }
        case LayerType::ConvertFp16ToFp32:
        {
            auto convertFp16ToFp32QueueDescriptor
                    = PolymorphicDowncast<const ConvertFp16ToFp32QueueDescriptor*>(&descriptor);
            return std::make_unique<RefConvertFp16ToFp32Workload>(*convertFp16ToFp32QueueDescriptor, info);
        }
        case LayerType::ConvertFp32ToFp16:
        {
            auto convertFp32ToFp16QueueDescriptor
                    = PolymorphicDowncast<const ConvertFp32ToFp16QueueDescriptor*>(&descriptor);
            return std::make_unique<RefConvertFp32ToFp16Workload>(*convertFp32ToFp16QueueDescriptor, info);
        }
        case LayerType::Convolution2d:
        {
            auto convolution2dQueueDescriptor = PolymorphicDowncast<const Convolution2dQueueDescriptor*>(&descriptor);
            return std::make_unique<RefConvolution2dWorkload>(*convolution2dQueueDescriptor, info);
        }
        case LayerType::Convolution3d:
        {
            auto convolution3dQueueDescriptor = PolymorphicDowncast<const Convolution3dQueueDescriptor*>(&descriptor);
            return std::make_unique<RefConvolution3dWorkload>(*convolution3dQueueDescriptor, info);
        }
        case LayerType::Debug:
        {
            auto debugQueueDescriptor = PolymorphicDowncast<const DebugQueueDescriptor*>(&descriptor);
            if (IsBFloat16(info))
            {
                return std::make_unique<RefDebugBFloat16Workload>(*debugQueueDescriptor, info);
            }
            if (IsFloat16(info))
            {
                return std::make_unique<RefDebugFloat16Workload>(*debugQueueDescriptor, info);
            }
            if (IsQSymmS16(info))
            {
                return std::make_unique<RefDebugQSymmS16Workload>(*debugQueueDescriptor, info);
            }
            if (IsQSymmS8(info))
            {
                return std::make_unique<RefDebugQSymmS8Workload>(*debugQueueDescriptor, info);
            }
            if (IsQAsymmU8(info))
            {
                return std::make_unique<RefDebugQAsymmU8Workload>(*debugQueueDescriptor, info);
            }
            if (IsQAsymmS8(info))
            {
                return std::make_unique<RefDebugQAsymmS8Workload>(*debugQueueDescriptor, info);
            }
            if (IsSigned32(info))
            {
                return std::make_unique<RefDebugSigned32Workload>(*debugQueueDescriptor, info);
            }

            return MakeWorkload<RefDebugFloat32Workload, RefDebugQAsymmU8Workload>(*debugQueueDescriptor, info);
        }
        case LayerType::DepthToSpace:
        {
            auto depthToSpaceQueueDescriptor = PolymorphicDowncast<const DepthToSpaceQueueDescriptor*>(&descriptor);
            return std::make_unique<RefDepthToSpaceWorkload>(*depthToSpaceQueueDescriptor, info);
        }
        case LayerType::DepthwiseConvolution2d:
        {
            auto depthwiseConvolution2DQueueDescriptor
                = PolymorphicDowncast<const DepthwiseConvolution2dQueueDescriptor*>(&descriptor);
            return std::make_unique<RefDepthwiseConvolution2dWorkload>(*depthwiseConvolution2DQueueDescriptor, info);
        }
        case LayerType::Dequantize:
        {
            auto dequantizeQueueDescriptor = PolymorphicDowncast<const DequantizeQueueDescriptor*>(&descriptor);
            return std::make_unique<RefDequantizeWorkload>(*dequantizeQueueDescriptor, info);
        }
        case LayerType::DetectionPostProcess:
        {
            auto detectionPostProcessQueueDescriptor
                = PolymorphicDowncast<const DetectionPostProcessQueueDescriptor*>(&descriptor);
            return std::make_unique<RefDetectionPostProcessWorkload>(*detectionPostProcessQueueDescriptor, info);
        }
        case LayerType::Division:
        {
            auto divisionQueueDescriptor = PolymorphicDowncast<const DivisionQueueDescriptor*>(&descriptor);
            if (info.m_InputTensorInfos[0].GetDataType() == armnn::DataType::Signed32)
            {
                return std::make_unique<RefDivisionWorkload<int32_t>>(*divisionQueueDescriptor, info);
            }
            else
            {
                return std::make_unique<RefDivisionWorkload<float>>(*divisionQueueDescriptor, info);
            }
        }
        case LayerType::ElementwiseBinary:
        {
            auto elementwiseBinaryQueueDescriptor
                    = PolymorphicDowncast<const ElementwiseBinaryQueueDescriptor*>(&descriptor);
            return std::make_unique<RefElementwiseBinaryWorkload>(*elementwiseBinaryQueueDescriptor, info);
        }
        case LayerType::ElementwiseUnary:
        {
            auto elementwiseUnaryQueueDescriptor
                = PolymorphicDowncast<const ElementwiseUnaryQueueDescriptor*>(&descriptor);
            if ((*elementwiseUnaryQueueDescriptor).m_Parameters.m_Operation == UnaryOperation::LogicalNot)
            {
                return std::make_unique<RefLogicalUnaryWorkload>(*elementwiseUnaryQueueDescriptor, info);
            }
            return std::make_unique<RefElementwiseUnaryWorkload>(*elementwiseUnaryQueueDescriptor, info);
        }
        case LayerType::FakeQuantization:
        {
            auto fakeQuantizationQueueDescriptor
                = PolymorphicDowncast<const FakeQuantizationQueueDescriptor*>(&descriptor);
            return std::make_unique<RefFakeQuantizationFloat32Workload>(*fakeQuantizationQueueDescriptor, info);
        }
        case LayerType::Fill:
        {
            auto fillQueueDescriptor = PolymorphicDowncast<const FillQueueDescriptor*>(&descriptor);
            return std::make_unique<RefFillWorkload>(*fillQueueDescriptor, info);
        }
        case LayerType::Floor:
        {
            auto floorQueueDescriptor = PolymorphicDowncast<const FloorQueueDescriptor*>(&descriptor);
            if(IsQuantizedType(info.m_InputTensorInfos[0].GetDataType()))
            {
                return nullptr;
            }
            else
            {
                return std::make_unique<RefFloorWorkload>(*floorQueueDescriptor, info);
            }
        }
        case LayerType::FullyConnected:
        {
            auto fullyConnectedQueueDescriptor
                    = PolymorphicDowncast<const FullyConnectedQueueDescriptor*>(&descriptor);
            return std::make_unique<RefFullyConnectedWorkload>(*fullyConnectedQueueDescriptor, info);
        }
        case LayerType::Gather:
        {
            auto gatherQueueDescriptor = PolymorphicDowncast<const GatherQueueDescriptor*>(&descriptor);
            return std::make_unique<RefGatherWorkload>(*gatherQueueDescriptor, info);
        }
        case LayerType::GatherNd:
        {
            auto gatherNdQueueDescriptor = PolymorphicDowncast<const GatherNdQueueDescriptor*>(&descriptor);
            return std::make_unique<RefGatherNdWorkload>(*gatherNdQueueDescriptor, info);
        }
        case LayerType::Input:
        {
            auto inputQueueDescriptor = PolymorphicDowncast<const InputQueueDescriptor*>(&descriptor);
            if (info.m_InputTensorInfos.empty() )
            {
                throw InvalidArgumentException("RefWorkloadFactory::CreateInput: Input cannot be zero length");
            }
            if (info.m_OutputTensorInfos.empty())
            {
                throw InvalidArgumentException("RefWorkloadFactory::CreateInput: Output cannot be zero length");
            }

            if (info.m_InputTensorInfos[0].GetNumBytes() != info.m_OutputTensorInfos[0].GetNumBytes())
            {
                throw InvalidArgumentException("RefWorkloadFactory::CreateInput: "
                                               "data input and output differ in byte count.");
            }

            return std::make_unique<CopyMemGenericWorkload>(*inputQueueDescriptor, info);
        }
        case LayerType::InstanceNormalization:
        {
            auto instanceNormalizationQueueDescriptor
                    = PolymorphicDowncast<const InstanceNormalizationQueueDescriptor*>(&descriptor);
            return std::make_unique<RefInstanceNormalizationWorkload>(*instanceNormalizationQueueDescriptor, info);
        }
        case LayerType::L2Normalization:
        {
            auto l2NormalizationQueueDescriptor
                    = PolymorphicDowncast<const L2NormalizationQueueDescriptor*>(&descriptor);
            return std::make_unique<RefL2NormalizationWorkload>(*l2NormalizationQueueDescriptor, info);
        }
        case LayerType::LogicalBinary:
        {
            auto logicalBinaryQueueDescriptor = PolymorphicDowncast<const LogicalBinaryQueueDescriptor*>(&descriptor);
            return std::make_unique<RefLogicalBinaryWorkload>(*logicalBinaryQueueDescriptor, info);
        }
        case LayerType::LogSoftmax:
        {
            auto logSoftmaxQueueDescriptor = PolymorphicDowncast<const LogSoftmaxQueueDescriptor*>(&descriptor);
            return std::make_unique<RefLogSoftmaxWorkload>(*logSoftmaxQueueDescriptor, info);
        }
        case LayerType::Lstm:
        {
            auto lstmQueueDescriptor = PolymorphicDowncast<const LstmQueueDescriptor*>(&descriptor);
            return std::make_unique<RefLstmWorkload>(*lstmQueueDescriptor, info);
        }
        case LayerType::Maximum:
        {
            auto maximumQueueDescriptor = PolymorphicDowncast<const MaximumQueueDescriptor*>(&descriptor);
            if (info.m_InputTensorInfos[0].GetDataType() == armnn::DataType::Signed32)
            {
                return std::make_unique<RefMaximumWorkload<int32_t>>(*maximumQueueDescriptor, info);
            }
            else
            {
                return std::make_unique<RefMaximumWorkload<float>>(*maximumQueueDescriptor, info);
            }
        }
        case LayerType::Mean:
        {
            auto meanQueueDescriptor = PolymorphicDowncast<const MeanQueueDescriptor*>(&descriptor);
            return  std::make_unique<RefMeanWorkload>(*meanQueueDescriptor, info);
        }
        case LayerType::MemCopy:
        {
            auto memCopyQueueDescriptor = PolymorphicDowncast<const MemCopyQueueDescriptor*>(&descriptor);
            if (descriptor.m_Inputs.empty())
            {
                throw InvalidArgumentException("RefWorkloadFactory: CreateMemCopy() expected an input tensor.");
            }
            return std::make_unique<CopyMemGenericWorkload>(*memCopyQueueDescriptor, info);
        }
        case LayerType::MemImport:
        {
            auto memImportQueueDescriptor = PolymorphicDowncast<const MemImportQueueDescriptor*>(&descriptor);
            if (descriptor.m_Inputs.empty())
            {
                throw InvalidArgumentException("RefWorkloadFactory: CreateMemImport() expected an input tensor.");
            }
            return std::make_unique<ImportMemGenericWorkload>(*memImportQueueDescriptor, info);
        }
        case LayerType::Minimum:
        {
            auto minimumQueueDescriptor = PolymorphicDowncast<const MinimumQueueDescriptor*>(&descriptor);
            if (info.m_InputTensorInfos[0].GetDataType() == armnn::DataType::Signed32)
            {
                return std::make_unique<RefMinimumWorkload<int32_t>>(*minimumQueueDescriptor, info);
            }
            else
            {
                return std::make_unique<RefMinimumWorkload<float>>(*minimumQueueDescriptor, info);
            }
        }
        case LayerType::Multiplication:
        {
            auto multiplicationQueueDescriptor
                    = PolymorphicDowncast<const MultiplicationQueueDescriptor*>(&descriptor);
            if (info.m_InputTensorInfos[0].GetDataType() == armnn::DataType::Signed32)
            {
                return std::make_unique<RefMultiplicationWorkload<int32_t>>(*multiplicationQueueDescriptor, info);
            }
            else
            {
                return std::make_unique<RefMultiplicationWorkload<float>>(*multiplicationQueueDescriptor, info);
            }
        }
        case LayerType::Normalization:
        {
            auto normalizationQueueDescriptor = PolymorphicDowncast<const NormalizationQueueDescriptor*>(&descriptor);
            return std::make_unique<RefNormalizationWorkload>(*normalizationQueueDescriptor, info);
        }
        case LayerType::Output:
        {
            auto outputQueueDescriptor = PolymorphicDowncast<const OutputQueueDescriptor*>(&descriptor);
            if (info.m_InputTensorInfos.empty() )
            {
                throw InvalidArgumentException("RefWorkloadFactory::CreateOutput: Input cannot be zero length");
            }
            if (info.m_OutputTensorInfos.empty())
            {
                throw InvalidArgumentException("RefWorkloadFactory::CreateOutput: Output cannot be zero length");
            }
            if (info.m_InputTensorInfos[0].GetNumBytes() != info.m_OutputTensorInfos[0].GetNumBytes())
            {
                throw InvalidArgumentException("RefWorkloadFactory::CreateOutput: data input and output "
                                               "differ in byte count.");
            }

            return std::make_unique<CopyMemGenericWorkload>(*outputQueueDescriptor, info);
        }
        case LayerType::Pad:
        {
            auto padQueueDescriptor = PolymorphicDowncast<const PadQueueDescriptor*>(&descriptor);
            return std::make_unique<RefPadWorkload>(*padQueueDescriptor, info);
        }
        case LayerType::Permute:
        {
            auto permuteQueueDescriptor = PolymorphicDowncast<const PermuteQueueDescriptor*>(&descriptor);
            if (IsQSymmS16(info))
            {
                return std::make_unique<RefPermuteQSymm16Workload>(*permuteQueueDescriptor, info);
            }
            else if (IsBFloat16(info))
            {
                return std::make_unique<RefPermuteBFloat16Workload>(*permuteQueueDescriptor, info);
            }
            else if (IsQAsymmS8(info))
            {
                return std::make_unique<RefPermuteQAsymmS8Workload>(*permuteQueueDescriptor, info);
            }
            return MakeWorkloadHelper<RefPermuteFloat16Workload, RefPermuteFloat32Workload, RefPermuteQAsymm8Workload,
                    NullWorkload, NullWorkload, NullWorkload>(*permuteQueueDescriptor, info);
        }
        case LayerType::Pooling2d:
        {
            auto pooling2dQueueDescriptor = PolymorphicDowncast<const Pooling2dQueueDescriptor*>(&descriptor);
            return std::make_unique<RefPooling2dWorkload>(*pooling2dQueueDescriptor, info);
        }
        case LayerType::Pooling3d:
        {
            auto pooling3dQueueDescriptor = PolymorphicDowncast<const Pooling3dQueueDescriptor*>(&descriptor);
            return std::make_unique<RefPooling3dWorkload>(*pooling3dQueueDescriptor, info);
        }
        case LayerType::PreCompiled:
        {
            return nullptr;
        }
        case LayerType::Prelu:
        {
            auto preluQueueDescriptor = PolymorphicDowncast<const PreluQueueDescriptor*>(&descriptor);
            return std::make_unique<RefPreluWorkload>(*preluQueueDescriptor, info);
        }
        case LayerType::QLstm:
        {
            auto qlstmQueueDescriptor = PolymorphicDowncast<const QLstmQueueDescriptor*>(&descriptor);
            return std::make_unique<RefQLstmWorkload>(*qlstmQueueDescriptor, info);
        }
        case LayerType::Quantize:
        {
            auto quantizeQueueDescriptor = PolymorphicDowncast<const QuantizeQueueDescriptor*>(&descriptor);
            return std::make_unique<RefQuantizeWorkload>(*quantizeQueueDescriptor, info);
        }
        case LayerType::Rank:
        {
            auto rankQueueDescriptor = PolymorphicDowncast<const RankQueueDescriptor*>(&descriptor);
            return std::make_unique<RefRankWorkload>(*rankQueueDescriptor, info);
        }
        case LayerType::Reduce:
        {
            auto reduceQueueDescriptor = PolymorphicDowncast<const ReduceQueueDescriptor*>(&descriptor);
            return std::make_unique<RefReduceWorkload>(*reduceQueueDescriptor, info);
        }
        case LayerType::Reshape:
        {
            auto reshapeQueueDescriptor = PolymorphicDowncast<const ReshapeQueueDescriptor*>(&descriptor);
            return std::make_unique<RefReshapeWorkload>(*reshapeQueueDescriptor, info);
        }
        case LayerType::Resize:
        {
            auto resizeQueueDescriptor = PolymorphicDowncast<const ResizeQueueDescriptor*>(&descriptor);
            return std::make_unique<RefResizeWorkload>(*resizeQueueDescriptor, info);
        }
        case LayerType::Shape:
        {
            auto shapeQueueDescriptor = PolymorphicDowncast<const ShapeQueueDescriptor*>(&descriptor);
            return std::make_unique<RefShapeWorkload>(*shapeQueueDescriptor, info);
        }
        case LayerType::Slice:
        {
            auto sliceQueueDescriptor = PolymorphicDowncast<const SliceQueueDescriptor*>(&descriptor);
            return std::make_unique<RefSliceWorkload>(*sliceQueueDescriptor, info);
        }
        case LayerType::Softmax:
        {
            auto softmaxQueueDescriptor = PolymorphicDowncast<const SoftmaxQueueDescriptor*>(&descriptor);
            return std::make_unique<RefSoftmaxWorkload>(*softmaxQueueDescriptor, info);
        }
        case LayerType::SpaceToBatchNd:
        {
            auto spaceToBatchNdQueueDescriptor
                    = PolymorphicDowncast<const SpaceToBatchNdQueueDescriptor*>(&descriptor);
            return std::make_unique<RefSpaceToBatchNdWorkload>(*spaceToBatchNdQueueDescriptor, info);
        }
        case LayerType::SpaceToDepth:
        {
            auto spaceToDepthQueueDescriptor = PolymorphicDowncast<const SpaceToDepthQueueDescriptor*>(&descriptor);
            return std::make_unique<RefSpaceToDepthWorkload>(*spaceToDepthQueueDescriptor, info);
        }
        case LayerType::Splitter:
        {
            auto splitterQueueDescriptor = PolymorphicDowncast<const SplitterQueueDescriptor*>(&descriptor);
            return std::make_unique<RefSplitterWorkload>(*splitterQueueDescriptor, info);
        }
        case LayerType::Stack:
        {
            auto stackQueueDescriptor = PolymorphicDowncast<const StackQueueDescriptor*>(&descriptor);
            return std::make_unique<RefStackWorkload>(*stackQueueDescriptor, info);
        }
        case LayerType::StridedSlice:
        {
            auto stridedSliceQueueDescriptor = PolymorphicDowncast<const StridedSliceQueueDescriptor*>(&descriptor);
            return std::make_unique<RefStridedSliceWorkload>(*stridedSliceQueueDescriptor, info);
        }
        case LayerType::Subtraction:
        {
            auto subtractionQueueDescriptor = PolymorphicDowncast<const SubtractionQueueDescriptor*>(&descriptor);
            if (info.m_InputTensorInfos[0].GetDataType() == armnn::DataType::Signed32)
            {
                return std::make_unique<RefSubtractionWorkload<int32_t>>(*subtractionQueueDescriptor, info);
            }
            else
            {
                return std::make_unique<RefSubtractionWorkload<float>>(*subtractionQueueDescriptor, info);
            }
        }
        case LayerType::Transpose:
        {
            auto transposeQueueDescriptor = PolymorphicDowncast<const TransposeQueueDescriptor*>(&descriptor);
            if (IsQSymmS16(info))
            {
                return std::make_unique<RefTransposeQSymm16Workload>(*transposeQueueDescriptor, info);
            }
            else if (IsBFloat16(info))
            {
                return std::make_unique<RefTransposeBFloat16Workload>(*transposeQueueDescriptor, info);
            }
            else if (IsQAsymmS8(info))
            {
                return std::make_unique<RefTransposeQAsymmS8Workload>(*transposeQueueDescriptor, info);
            }
            return MakeWorkloadHelper<RefTransposeFloat16Workload, RefTransposeFloat32Workload,
                    RefTransposeQAsymm8Workload, NullWorkload, NullWorkload, NullWorkload>
                    (*transposeQueueDescriptor, info);
        }
        case LayerType::TransposeConvolution2d:
        {
            auto transposeConvolution2dQueueDescriptor
                    = PolymorphicDowncast<const TransposeConvolution2dQueueDescriptor*>(&descriptor);
            return std::make_unique<RefTransposeConvolution2dWorkload>(*transposeConvolution2dQueueDescriptor, info);
        }
        case LayerType::UnidirectionalSequenceLstm:
        {
            auto unidirectionalSequenceLstmQueueDescriptor
                    = PolymorphicDowncast<const UnidirectionalSequenceLstmQueueDescriptor*>(&descriptor);
            return std::make_unique<RefUnidirectionalSequenceLstmWorkload>(*unidirectionalSequenceLstmQueueDescriptor,
                                                                           info);
        }
        default:
            return nullptr;
    }
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateActivation(const ActivationQueueDescriptor& descriptor,
                                                                const WorkloadInfo& info) const
{
    return std::make_unique<RefActivationWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    if (info.m_InputTensorInfos[0].GetDataType() == armnn::DataType::Signed32)
    {
        return std::make_unique<RefAdditionWorkload<int32_t>>(descriptor, info);
    }
    else
    {
        return std::make_unique<RefAdditionWorkload<float>>(descriptor, info);
    }
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateArgMinMax(const ArgMinMaxQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    return std::make_unique<RefArgMinMaxWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateBatchNormalization(
    const BatchNormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<RefBatchNormalizationWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateBatchToSpaceNd(const BatchToSpaceNdQueueDescriptor& descriptor,
                                                                    const WorkloadInfo& info) const
{
    return std::make_unique<RefBatchToSpaceNdWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateCast(const CastQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return std::make_unique<RefCastWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateChannelShuffle(const ChannelShuffleQueueDescriptor &descriptor,
                                                                    const WorkloadInfo &info) const
{
    return std::make_unique<RefChannelShuffleWorkload>(descriptor,info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateComparison(const ComparisonQueueDescriptor& descriptor,
                                                                const WorkloadInfo& info) const
{
    return std::make_unique<RefComparisonWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateConcat(const ConcatQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return std::make_unique<RefConcatWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateConstant(const ConstantQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    return std::make_unique<RefConstantWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateConvertFp16ToFp32(
    const ConvertFp16ToFp32QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<RefConvertFp16ToFp32Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateConvertFp32ToFp16(
    const ConvertFp32ToFp16QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<RefConvertFp32ToFp16Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateConvolution2d(const Convolution2dQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const
{
    return std::make_unique<RefConvolution2dWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateConvolution3d(const Convolution3dQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const
{
    return std::make_unique<RefConvolution3dWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateDebug(const DebugQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    if (IsBFloat16(info))
    {
        return std::make_unique<RefDebugBFloat16Workload>(descriptor, info);
    }
    if (IsFloat16(info))
    {
        return std::make_unique<RefDebugFloat16Workload>(descriptor, info);
    }
    if (IsQSymmS16(info))
    {
        return std::make_unique<RefDebugQSymmS16Workload>(descriptor, info);
    }
    if (IsQSymmS8(info))
    {
        return std::make_unique<RefDebugQSymmS8Workload>(descriptor, info);
    }
    if (IsQAsymmU8(info))
    {
        return std::make_unique<RefDebugQAsymmU8Workload>(descriptor, info);
    }
    if (IsQAsymmS8(info))
    {
        return std::make_unique<RefDebugQAsymmS8Workload>(descriptor, info);
    }
    if (IsSigned32(info))
    {
        return std::make_unique<RefDebugSigned32Workload>(descriptor, info);
    }

    return MakeWorkload<RefDebugFloat32Workload, RefDebugQAsymmU8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateDepthToSpace(const DepthToSpaceQueueDescriptor& descriptor,
                                                                  const WorkloadInfo& info) const
{
    return std::make_unique<RefDepthToSpaceWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateDepthwiseConvolution2d(
    const DepthwiseConvolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<RefDepthwiseConvolution2dWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateDequantize(const DequantizeQueueDescriptor& descriptor,
                                                                const WorkloadInfo& info) const
{
    return std::make_unique<RefDequantizeWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateDetectionPostProcess(
    const DetectionPostProcessQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<RefDetectionPostProcessWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateDivision(const DivisionQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    if (info.m_InputTensorInfos[0].GetDataType() == armnn::DataType::Signed32)
    {
        return std::make_unique<RefDivisionWorkload<int32_t>>(descriptor, info);
    }
    else
    {
        return std::make_unique<RefDivisionWorkload<float>>(descriptor, info);
    }
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateElementwiseUnary(const ElementwiseUnaryQueueDescriptor& descriptor,
                                                                      const WorkloadInfo& info) const
{
    if (descriptor.m_Parameters.m_Operation == UnaryOperation::LogicalNot)
    {
        return std::make_unique<RefLogicalUnaryWorkload>(descriptor, info);
    }
    return std::make_unique<RefElementwiseUnaryWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateFakeQuantization(const FakeQuantizationQueueDescriptor& descriptor,
                                                                      const WorkloadInfo& info) const
{
    return MakeWorkload<RefFakeQuantizationFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateFill(const FillQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return std::make_unique<RefFillWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateFloor(const FloorQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    if(IsQuantizedType(info.m_InputTensorInfos[0].GetDataType()))
    {
        return nullptr;
    }
    else
    {
        return std::make_unique<RefFloorWorkload>(descriptor, info);
    }
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateFullyConnected(
    const FullyConnectedQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<RefFullyConnectedWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateGather(const GatherQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return std::make_unique<RefGatherWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    if (info.m_InputTensorInfos.empty() )
    {
        throw InvalidArgumentException("RefWorkloadFactory::CreateInput: Input cannot be zero length");
    }
    if (info.m_OutputTensorInfos.empty())
    {
        throw InvalidArgumentException("RefWorkloadFactory::CreateInput: Output cannot be zero length");
    }

    if (info.m_InputTensorInfos[0].GetNumBytes() != info.m_OutputTensorInfos[0].GetNumBytes())
    {
        throw InvalidArgumentException("RefWorkloadFactory::CreateInput: data input and output differ in byte count.");
    }

    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateInstanceNormalization(
    const InstanceNormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<RefInstanceNormalizationWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateL2Normalization(const L2NormalizationQueueDescriptor& descriptor,
                                                                     const WorkloadInfo& info) const
{
    return std::make_unique<RefL2NormalizationWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateLogicalBinary(const LogicalBinaryQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const
{
    return std::make_unique<RefLogicalBinaryWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateLogSoftmax(const LogSoftmaxQueueDescriptor& descriptor,
                                                                const WorkloadInfo& info) const
{
    return std::make_unique<RefLogSoftmaxWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateLstm(const LstmQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return std::make_unique<RefLstmWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateMaximum(const MaximumQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    if (info.m_InputTensorInfos[0].GetDataType() == armnn::DataType::Signed32)
    {
        return std::make_unique<RefMaximumWorkload<int32_t>>(descriptor, info);
    }
    else
    {
        return std::make_unique<RefMaximumWorkload<float>>(descriptor, info);
    }
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateMean(const MeanQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return  std::make_unique<RefMeanWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    if (descriptor.m_Inputs.empty())
    {
        throw InvalidArgumentException("RefWorkloadFactory: CreateMemCopy() expected an input tensor.");
    }
    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateMemImport(const MemImportQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    if (descriptor.m_Inputs.empty())
    {
        throw InvalidArgumentException("RefWorkloadFactory: CreateMemImport() expected an input tensor.");
    }
    return std::make_unique<ImportMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateMinimum(const MinimumQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    if (info.m_InputTensorInfos[0].GetDataType() == armnn::DataType::Signed32)
    {
        return std::make_unique<RefMinimumWorkload<int32_t>>(descriptor, info);
    }
    else
    {
        return std::make_unique<RefMinimumWorkload<float>>(descriptor, info);
    }
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateMultiplication(const MultiplicationQueueDescriptor& descriptor,
                                                                    const WorkloadInfo& info) const
{
    if (info.m_InputTensorInfos[0].GetDataType() == armnn::DataType::Signed32)
    {
        return std::make_unique<RefMultiplicationWorkload<int32_t>>(descriptor, info);
    }
    else
    {
        return std::make_unique<RefMultiplicationWorkload<float>>(descriptor, info);
    }
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateNormalization(const NormalizationQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const
{
    return std::make_unique<RefNormalizationWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateOutput(const OutputQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    if (info.m_InputTensorInfos.empty() )
    {
        throw InvalidArgumentException("RefWorkloadFactory::CreateOutput: Input cannot be zero length");
    }
    if (info.m_OutputTensorInfos.empty())
    {
        throw InvalidArgumentException("RefWorkloadFactory::CreateOutput: Output cannot be zero length");
    }
    if (info.m_InputTensorInfos[0].GetNumBytes() != info.m_OutputTensorInfos[0].GetNumBytes())
    {
        throw InvalidArgumentException("RefWorkloadFactory::CreateOutput: data input and output differ in byte count.");
    }

    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreatePad(const PadQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info) const
{
    return std::make_unique<RefPadWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreatePermute(const PermuteQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    if (IsQSymmS16(info))
    {
        return std::make_unique<RefPermuteQSymm16Workload>(descriptor, info);
    }
    else if (IsBFloat16(info))
    {
        return std::make_unique<RefPermuteBFloat16Workload>(descriptor, info);
    }
    else if (IsQAsymmS8(info))
    {
        return std::make_unique<RefPermuteQAsymmS8Workload>(descriptor, info);
    }
    return MakeWorkloadHelper<RefPermuteFloat16Workload, RefPermuteFloat32Workload, RefPermuteQAsymm8Workload,
        NullWorkload, NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreatePooling2d(const Pooling2dQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    return std::make_unique<RefPooling2dWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreatePooling3d(const Pooling3dQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    return std::make_unique<RefPooling3dWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreatePreCompiled(const PreCompiledQueueDescriptor& /*descriptor*/,
                                                                 const WorkloadInfo& /*info*/) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreatePrelu(const PreluQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return std::make_unique<RefPreluWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateQLstm(const QLstmQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return std::make_unique<RefQLstmWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateQuantize(const QuantizeQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    return std::make_unique<RefQuantizeWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateRank(const RankQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return std::make_unique<RefRankWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateReduce(const ReduceQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return std::make_unique<RefReduceWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateReshape(const ReshapeQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    return std::make_unique<RefReshapeWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateResize(const ResizeQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return std::make_unique<RefResizeWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateShape(const ShapeQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return std::make_unique<RefShapeWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateSlice(const SliceQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return std::make_unique<RefSliceWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateSoftmax(const SoftmaxQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    return std::make_unique<RefSoftmaxWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateSpaceToBatchNd(const SpaceToBatchNdQueueDescriptor& descriptor,
                                                                    const WorkloadInfo& info) const
{
    return std::make_unique<RefSpaceToBatchNdWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateSpaceToDepth(const SpaceToDepthQueueDescriptor& descriptor,
                                                                  const WorkloadInfo& info) const
{
    return std::make_unique<RefSpaceToDepthWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateSplitter(const SplitterQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    return std::make_unique<RefSplitterWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateStack(const StackQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return std::make_unique<RefStackWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateStridedSlice(const StridedSliceQueueDescriptor& descriptor,
                                                                  const WorkloadInfo& info) const
{
    return std::make_unique<RefStridedSliceWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateSubtraction(const SubtractionQueueDescriptor& descriptor,
                                                                 const WorkloadInfo& info) const
{
    if (info.m_InputTensorInfos[0].GetDataType() == armnn::DataType::Signed32)
    {
        return std::make_unique<RefSubtractionWorkload<int32_t>>(descriptor, info);
    }
    else
    {
        return std::make_unique<RefSubtractionWorkload<float>>(descriptor, info);
    }
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateTranspose(const TransposeQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    if (IsQSymmS16(info))
    {
        return std::make_unique<RefTransposeQSymm16Workload>(descriptor, info);
    }
    else if (IsBFloat16(info))
    {
        return std::make_unique<RefTransposeBFloat16Workload>(descriptor, info);
    }
    else if (IsQAsymmS8(info))
    {
        return std::make_unique<RefTransposeQAsymmS8Workload>(descriptor, info);
    }
    return MakeWorkloadHelper<RefTransposeFloat16Workload, RefTransposeFloat32Workload, RefTransposeQAsymm8Workload,
            NullWorkload, NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateTransposeConvolution2d(
    const TransposeConvolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<RefTransposeConvolution2dWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateUnidirectionalSequenceLstm(
    const UnidirectionalSequenceLstmQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<RefUnidirectionalSequenceLstmWorkload>(descriptor, info);;
}

} // namespace armnn
