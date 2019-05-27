//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/MemCopyWorkload.hpp>
#include <backendsCommon/MakeWorkloadHelper.hpp>
#include "RefWorkloadFactory.hpp"
#include "RefBackendId.hpp"
#include "workloads/RefWorkloads.hpp"
#include "Layer.hpp"

#include <boost/log/trivial.hpp>

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
    return armnn::MakeWorkloadHelper<NullWorkload, F32Workload, U8Workload, NullWorkload, NullWorkload>(descriptor,
                                                                                                        info);
}

bool IsFloat16(const WorkloadInfo& info)
{
    auto checkFloat16 = [](const TensorInfo& tensorInfo) {return tensorInfo.GetDataType() == DataType::Float16;};
    auto it = std::find_if(std::begin(info.m_InputTensorInfos), std::end(info.m_InputTensorInfos), checkFloat16);
    if (it != std::end(info.m_InputTensorInfos))
    {
        return true;
    }
    it = std::find_if(std::begin(info.m_OutputTensorInfos), std::end(info.m_OutputTensorInfos), checkFloat16);
    if (it != std::end(info.m_OutputTensorInfos))
    {
        return true;
    }
    return false;
}

RefWorkloadFactory::RefWorkloadFactory()
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

std::unique_ptr<ITensorHandle> RefWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    return std::make_unique<ScopedCpuTensorHandle>(tensorInfo);
}

std::unique_ptr<ITensorHandle> RefWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                      DataLayout dataLayout) const
{
    return std::make_unique<ScopedCpuTensorHandle>(tensorInfo);
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

    return MakeWorkload<CopyMemGenericWorkload, CopyMemGenericWorkload>(descriptor, info);
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

    return MakeWorkloadHelper<CopyMemGenericWorkload, CopyMemGenericWorkload,
                              CopyMemGenericWorkload, NullWorkload, CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateActivation(const ActivationQueueDescriptor& descriptor,
                                                                const WorkloadInfo&              info) const
{
    if (IsFloat16(info))
    {
        return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
    }
    return std::make_unique<RefActivationWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateSoftmax(const SoftmaxQueueDescriptor& descriptor,
                                                             const WorkloadInfo&           info) const
{
    return MakeWorkload<RefSoftmaxFloat32Workload, RefSoftmaxUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateSplitter(const SplitterQueueDescriptor& descriptor,
                                                              const WorkloadInfo&            info) const
{
    return MakeWorkload<RefSplitterFloat32Workload, RefSplitterUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateMerger(const MergerQueueDescriptor& descriptor,
                                                                   const WorkloadInfo&          info) const
{
    return CreateConcat(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateFullyConnected(
    const FullyConnectedQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return std::make_unique<RefFullyConnectedWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreatePermute(const PermuteQueueDescriptor& descriptor,
                                                                    const WorkloadInfo&           info) const
{
    return MakeWorkloadHelper<RefPermuteFloat16Workload, RefPermuteFloat32Workload, RefPermuteUint8Workload,
        NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreatePooling2d(const Pooling2dQueueDescriptor& descriptor,
                                                                      const WorkloadInfo&           info) const
{
    return MakeWorkload<RefPooling2dFloat32Workload, RefPooling2dUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateConvolution2d(
    const Convolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return std::make_unique<RefConvolution2dWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateDepthwiseConvolution2d(
    const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return std::make_unique<RefDepthwiseConvolution2dWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateDetectionPostProcess(
    const armnn::DetectionPostProcessQueueDescriptor& descriptor, const armnn::WorkloadInfo& info) const
{
    const DataType dataType = info.m_InputTensorInfos[0].GetDataType();
    switch (dataType)
    {
        case DataType::Float32:
            return std::make_unique<RefDetectionPostProcessFloat32Workload>(descriptor, info);
        case DataType::QuantisedAsymm8:
            return std::make_unique<RefDetectionPostProcessUint8Workload>(descriptor, info);
        default:
            return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
    }
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateNormalization(
    const NormalizationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<RefNormalizationFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                                     const WorkloadInfo&            info) const
{
    if (IsFloat16(info))
    {
        return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
    }
    return std::make_unique<RefAdditionWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateMultiplication(
    const MultiplicationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    if (IsFloat16(info))
    {
        return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
    }
    return std::make_unique<RefMultiplicationWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateBatchNormalization(
    const BatchNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<RefBatchNormalizationFloat32Workload, RefBatchNormalizationUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                                                    const WorkloadInfo&        info) const
{
    if (descriptor.m_Inputs.empty())
    {
        throw InvalidArgumentException("RefWorkloadFactory: CreateMemCopy() expected an input tensor.");
    }
    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateResizeBilinear(const ResizeBilinearQueueDescriptor& descriptor,
                                                                    const WorkloadInfo& info) const
{
    return MakeWorkload<RefResizeBilinearFloat32Workload, RefResizeBilinearUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateFakeQuantization(
    const FakeQuantizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<RefFakeQuantizationFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateL2Normalization(const L2NormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<RefL2NormalizationFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateConcat(const MergerQueueDescriptor& descriptor,
                                                                   const WorkloadInfo&          info) const
{
    if (IsFloat16(info))
    {
        return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
    }
    return std::make_unique<RefConcatWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateConstant(const ConstantQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<RefConstantWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateReshape(const ReshapeQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<RefReshapeWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateSpaceToBatchNd(const SpaceToBatchNdQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<RefSpaceToBatchNdFloat32Workload, RefSpaceToBatchNdUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateFloor(const FloorQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return MakeWorkload<RefFloorFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateLstm(const LstmQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<RefLstmWorkload>(descriptor, info);
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

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateDivision(
    const DivisionQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    if (IsFloat16(info))
    {
        return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
    }
    return std::make_unique<RefDivisionWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateSubtraction(
    const SubtractionQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    if (IsFloat16(info))
    {
        return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
    }
    return std::make_unique<RefSubtractionWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateMaximum(
    const MaximumQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return std::make_unique<RefMaximumWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateMean(
    const MeanQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<RefMeanFloat32Workload, RefMeanUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateMinimum(
    const MinimumQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return std::make_unique<RefMinimumWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreatePad(const PadQueueDescriptor& descriptor,
                                                 const WorkloadInfo& info) const
{
    return MakeWorkload<RefPadFloat32Workload, RefPadUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateEqual(const EqualQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return std::make_unique<RefEqualWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateBatchToSpaceNd(const BatchToSpaceNdQueueDescriptor& descriptor,
                                                                    const WorkloadInfo& info) const
{
    return MakeWorkload<RefBatchToSpaceNdFloat32Workload, RefBatchToSpaceNdUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateStridedSlice(const StridedSliceQueueDescriptor& descriptor,
                                                                  const WorkloadInfo& info) const
{
    return MakeWorkload<RefStridedSliceFloat32Workload, RefStridedSliceUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateGreater(const GreaterQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    return std::make_unique<RefGreaterWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateDebug(const DebugQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return MakeWorkload<RefDebugFloat32Workload, RefDebugUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateRsqrt(const RsqrtQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return MakeWorkload<RefRsqrtFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateGather(const armnn::GatherQueueDescriptor& descriptor,
                                                            const armnn::WorkloadInfo& info) const
{
    return MakeWorkload<RefGatherFloat32Workload, RefGatherUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreatePreCompiled(const PreCompiledQueueDescriptor& descriptor,
                                                                 const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateQuantize(const QuantizeQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    return std::make_unique<RefQuantizeWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateDequantize(const DequantizeQueueDescriptor& descriptor,
                                                                const WorkloadInfo& info) const
{
    return std::make_unique<RefDequantizeWorkload>(descriptor, info);
}

} // namespace armnn
