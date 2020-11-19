//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <Layer.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/MemCopyWorkload.hpp>
#include <backendsCommon/MemImportWorkload.hpp>
#include <backendsCommon/MakeWorkloadHelper.hpp>
#include <reference/workloads/RefFillWorkload.hpp>
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
    // For Ref it is okay to make the TensorHandle memory managed as it can also store a pointer
    // to unmanaged memory. This also ensures memory alignment.
    IgnoreUnused(isMemoryManaged);
    return std::make_unique<RefTensorHandle>(tensorInfo, m_MemoryManager);
}

std::unique_ptr<ITensorHandle> RefWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                      DataLayout dataLayout,
                                                                      const bool isMemoryManaged) const
{
    // For Ref it is okay to make the TensorHandle memory managed as it can also store a pointer
    // to unmanaged memory. This also ensures memory alignment.
    IgnoreUnused(isMemoryManaged, dataLayout);
    return std::make_unique<RefTensorHandle>(tensorInfo, m_MemoryManager);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateAbs(const AbsQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info) const
{
    IgnoreUnused(descriptor);
    ElementwiseUnaryQueueDescriptor elementwiseUnaryDescriptor;
    elementwiseUnaryDescriptor.m_Parameters.m_Operation = UnaryOperation::Abs;

    return CreateElementwiseUnary(elementwiseUnaryDescriptor, info);
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

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateConvertBf16ToFp32(
    const ConvertBf16ToFp32QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<RefConvertBf16ToFp32Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateConvertFp16ToFp32(
    const ConvertFp16ToFp32QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<RefConvertFp16ToFp32Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateConvertFp32ToBf16(
    const ConvertFp32ToBf16QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<RefConvertFp32ToBf16Workload>(descriptor, info);
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

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateEqual(const EqualQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    IgnoreUnused(descriptor);
    ComparisonQueueDescriptor comparisonDescriptor;
    comparisonDescriptor.m_Parameters.m_Operation = ComparisonOperation::Equal;

    return CreateComparison(comparisonDescriptor, info);
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

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateGreater(const GreaterQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    IgnoreUnused(descriptor);
    ComparisonQueueDescriptor comparisonDescriptor;
    comparisonDescriptor.m_Parameters.m_Operation = ComparisonOperation::Greater;

    return CreateComparison(comparisonDescriptor, info);
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

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateMerger(const MergerQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return CreateConcat(descriptor, info);
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

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateResizeBilinear(const ResizeBilinearQueueDescriptor& descriptor,
                                                                    const WorkloadInfo& info) const
{
    ResizeQueueDescriptor resizeDescriptor;
    resizeDescriptor.m_Parameters.m_Method       = ResizeMethod::Bilinear;
    resizeDescriptor.m_Parameters.m_DataLayout   = descriptor.m_Parameters.m_DataLayout;
    resizeDescriptor.m_Parameters.m_TargetWidth  = descriptor.m_Parameters.m_TargetWidth;
    resizeDescriptor.m_Parameters.m_TargetHeight = descriptor.m_Parameters.m_TargetHeight;

    return CreateResize(resizeDescriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateRsqrt(const RsqrtQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    IgnoreUnused(descriptor);
    ElementwiseUnaryQueueDescriptor elementwiseUnaryDescriptor;
    elementwiseUnaryDescriptor.m_Parameters.m_Operation = UnaryOperation::Rsqrt;

    return CreateElementwiseUnary(elementwiseUnaryDescriptor, info);
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

} // namespace armnn
