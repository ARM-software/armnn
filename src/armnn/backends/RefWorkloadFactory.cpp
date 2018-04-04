//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "CpuTensorHandle.hpp"
#include "RefWorkloadFactory.hpp"
#include "RefWorkloads.hpp"
#include "Layer.hpp"
#include "Layers.hpp"
#include "MemCopyWorkload.hpp"
#include "MakeWorkloadHelper.hpp"

#include <boost/log/trivial.hpp>

namespace armnn
{

template <typename F32Workload, typename U8Workload, typename QueueDescriptorType>
std::unique_ptr<IWorkload> RefWorkloadFactory::MakeWorkload(const QueueDescriptorType& descriptor,
    const WorkloadInfo& info) const
{
    if (!IsOperationQueueDescriptor(descriptor) || m_OperationWorkloadsAllowed)
    {
        return armnn::MakeWorkload<F32Workload, U8Workload>(descriptor, info);
    }
    else
    {
        return std::unique_ptr<IWorkload>();
    }
}

RefWorkloadFactory::RefWorkloadFactory(bool operationWorkloadsAllowed)
    : m_OperationWorkloadsAllowed(operationWorkloadsAllowed)
{
}

bool RefWorkloadFactory::IsLayerSupported(const Layer& layer, DataType dataType, std::string& outReasonIfUnsupported)
{
    return IWorkloadFactory::IsLayerSupported(Compute::CpuRef, layer, dataType, outReasonIfUnsupported);
}

std::unique_ptr<ITensorHandle> RefWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
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

    return MakeWorkload<CopyFromCpuToCpuFloat32Workload, CopyFromCpuToCpuUint8Workload>(descriptor, info);
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

    return MakeWorkload<CopyFromCpuToCpuFloat32Workload, CopyFromCpuToCpuUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateActivation(const ActivationQueueDescriptor& descriptor,
                                                                const WorkloadInfo&              info) const
{
    return MakeWorkload<RefActivationFloat32Workload, RefActivationUint8Workload>(descriptor, info);
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
    return MakeWorkload<RefMergerFloat32Workload, RefMergerUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateFullyConnected(
    const FullyConnectedQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<RefFullyConnectedFloat32Workload, RefFullyConnectedUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreatePermute(const PermuteQueueDescriptor& descriptor,
                                                                    const WorkloadInfo&           info) const
{
    return MakeWorkload<RefPermuteFloat32Workload, RefPermuteUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreatePooling2d(const Pooling2dQueueDescriptor& descriptor,
                                                                      const WorkloadInfo&           info) const
{
    return MakeWorkload<RefPooling2dFloat32Workload, RefPooling2dUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateConvolution2d(
    const Convolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<RefConvolution2dFloat32Workload, RefConvolution2dUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateDepthwiseConvolution2d(
    const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<RefDepthwiseConvolution2dFloat32Workload,
        RefDepthwiseConvolution2dUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateNormalization(
    const NormalizationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<RefNormalizationFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                                     const WorkloadInfo&            info) const
{
    return MakeWorkload<RefAdditionFloat32Workload, RefAdditionUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> RefWorkloadFactory::CreateMultiplication(
    const MultiplicationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<RefMultiplicationFloat32Workload, RefMultiplicationUint8Workload>(descriptor, info);
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
    // Create a workload that will copy tensor data from the inputs, which can have a number of different formats,
    // to CPU tensors.
     switch (descriptor.m_Inputs[0]->GetType())
    {
#if ARMCOMPUTECL_ENABLED
    case ITensorHandle::CL:
    {
        return MakeWorkload<CopyFromClToCpuFloat32Workload, CopyFromClToCpuUint8Workload>(descriptor, info);
    }
#endif
#if ARMCOMPUTENEON_ENABLED
    case ITensorHandle::Neon:
    {
        return MakeWorkload<CopyFromNeonToCpuFloat32Workload, CopyFromNeonToCpuUint8Workload>(descriptor, info);
    }
#endif
    default:
        throw InvalidArgumentException("RefWorkloadFactory: Destination type not supported for MemCopy Workload.");
        return nullptr;
    }
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

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateConstant(const ConstantQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<RefConstantFloat32Workload, RefConstantUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateReshape(const ReshapeQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<RefReshapeFloat32Workload, RefReshapeUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> RefWorkloadFactory::CreateFloor(const FloorQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<RefFloorFloat32Workload, NullWorkload>(descriptor, info);
}

// for yolov2
std::unique_ptr<IWorkload> CreateDetectionOutput(const DetectionOutputQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info) const
{
    return MakeWorkload<RefFloorFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> CreateReorg(const ReorgQueueDescriptor& descriptor,
                                               const WorkloadInfo& info) const
{
    return MakeWorkload<RefFloorFloat32Workload, NullWorkload>(descriptor, info);
}

} // namespace armnn
