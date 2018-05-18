//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "NeonWorkloadFactory.hpp"
#include "armnn/Utils.hpp"
#include "CpuTensorHandle.hpp"
#include "Layer.hpp"

#ifdef ARMCOMPUTENEON_ENABLED
#include "arm_compute/runtime/Allocator.h"
#include "MemCopyWorkload.hpp"
#include "NeonTensorHandle.hpp"
#include "NeonWorkloadUtils.hpp"
#include "NeonWorkloads.hpp"
#endif

#include "MakeWorkloadHelper.hpp"

#include <boost/polymorphic_cast.hpp>

namespace armnn
{

bool NeonWorkloadFactory::IsLayerSupported(const Layer& layer, DataType dataType, std::string& outReasonIfUnsupported)
{
    return IWorkloadFactory::IsLayerSupported(Compute::CpuAcc, layer, dataType, outReasonIfUnsupported);
}

#ifdef ARMCOMPUTENEON_ENABLED

NeonWorkloadFactory::NeonWorkloadFactory()
: m_MemoryManager(std::make_unique<arm_compute::Allocator>())
{
}

std::unique_ptr<ITensorHandle> NeonWorkloadFactory::CreateSubTensorHandle(ITensorHandle& parent,
    TensorShape const& subTensorShape,
    unsigned int const* subTensorOrigin) const
{
    BOOST_ASSERT(parent.GetType() == ITensorHandle::Neon);

    const arm_compute::TensorShape shape = armcomputetensorutils::BuildArmComputeTensorShape(subTensorShape);

    arm_compute::Coordinates coords;
    coords.set_num_dimensions(subTensorShape.GetNumDimensions());
    for (unsigned int i = 0; i < subTensorShape.GetNumDimensions(); i++)
    {
        // arm compute indexes tensor coords in reverse order
        unsigned int revertedIndex = subTensorShape.GetNumDimensions() - i - 1;
        coords.set(i, boost::numeric_cast<int>(subTensorOrigin[revertedIndex]));
    }

    return std::make_unique<NeonSubTensorHandle>(boost::polymorphic_downcast<INeonTensorHandle*>(&parent)->GetTensor(),
        shape, coords);
}

std::unique_ptr<ITensorHandle> NeonWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    return std::make_unique<NeonTensorHandle>(tensorInfo);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                            const WorkloadInfo&        info) const
{
    return MakeWorkload<CopyFromCpuToNeonFloat32Workload, CopyFromCpuToNeonUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateOutput(const OutputQueueDescriptor& descriptor,
                                                             const WorkloadInfo&        info) const
{
    return MakeWorkload<CopyFromNeonToCpuFloat32Workload, CopyFromNeonToCpuUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateActivation(const ActivationQueueDescriptor& descriptor,
                                                                 const WorkloadInfo&              info) const
{
    return MakeWorkload<NeonActivationFloat32Workload, NeonActivationUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateSoftmax(const SoftmaxQueueDescriptor& descriptor,
                                                              const WorkloadInfo&           info) const
{
    return MakeWorkload<NeonSoftmaxFloat32Workload, NeonSoftmaxUint8Workload>(descriptor, info,
                                                                              m_MemoryManager.Get());
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateSplitter(const SplitterQueueDescriptor& descriptor,
                                                               const WorkloadInfo&            info) const
{
    return MakeWorkload<NeonSplitterFloat32Workload, NeonSplitterUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateMerger(const MergerQueueDescriptor& descriptor,
                                                                    const WorkloadInfo&          info) const
{
    return MakeWorkload<NeonMergerFloat32Workload, NeonMergerUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateFullyConnected(
    const FullyConnectedQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NeonFullyConnectedFloat32Workload, NullWorkload>(descriptor, info, m_MemoryManager.Get());
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreatePermute(const PermuteQueueDescriptor& descriptor,
                                                                     const WorkloadInfo&           info) const
{
    return MakeWorkload<NeonPermuteFloat32Workload, NeonPermuteUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreatePooling2d(const Pooling2dQueueDescriptor& descriptor,
                                                                       const WorkloadInfo&           info) const
{
    return MakeWorkload<NeonPooling2dFloat32Workload, NeonPooling2dUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateConvolution2d(
    const Convolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NeonConvolution2dFloat32Workload, NeonConvolution2dUint8Workload>(descriptor, info,
                                                                                          m_MemoryManager.Get());
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateDepthwiseConvolution2d(
    const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NeonDepthwiseConvolutionFloat32Workload, NeonDepthwiseConvolutionUint8Workload>(
        descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateNormalization(
    const NormalizationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NeonNormalizationFloat32Workload, NullWorkload>(descriptor, info, m_MemoryManager.Get());
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                                      const WorkloadInfo&            info) const
{
    return MakeWorkload<NeonAdditionFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateMultiplication(
    const MultiplicationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NeonMultiplicationFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateBatchNormalization(
    const BatchNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NeonBatchNormalizationFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                                                     const WorkloadInfo&        info) const
{
    if (descriptor.m_Inputs.empty() || !descriptor.m_Inputs[0])
    {
        throw InvalidArgumentException("NeonWorkloadFactory: Invalid null input for MemCopy workload");
    }

    // Create a workload that will copy tensor data from the inputs, which can have a number of different formats,
    // to Neon tensors.
    switch (descriptor.m_Inputs[0]->GetType())
    {
    case ITensorHandle::Cpu:
        return MakeWorkload<CopyFromCpuToNeonFloat32Workload, CopyFromCpuToNeonUint8Workload>(descriptor, info);
#if ARMCOMPUTECL_ENABLED
    case ITensorHandle::CL:
    {
        return MakeWorkload<CopyFromClToNeonFloat32Workload, CopyFromClToNeonUint8Workload>(descriptor, info);
    }
#endif
    default:
        throw InvalidArgumentException("NeonWorkloadFactory: Destination type not supported for MemCopy Workload.");
    }
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateResizeBilinear(
    const ResizeBilinearQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateFakeQuantization(
    const FakeQuantizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateL2Normalization(const L2NormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<NeonL2NormalizationFloat32Workload, NullWorkload>(descriptor, info, m_MemoryManager.Get());
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateConstant(const ConstantQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<NeonConstantFloat32Workload, NeonConstantUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateReshape(const ReshapeQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<NeonReshapeFloat32Workload, NeonReshapeUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateFloor(const FloorQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<NeonFloorFloat32Workload, NullWorkload>(descriptor, info);
}

void NeonWorkloadFactory::Finalize()
{
    m_MemoryManager.Finalize();
}

#else // Compiled without ArmCompute libs

NeonWorkloadFactory::NeonWorkloadFactory()
{
}

std::unique_ptr<ITensorHandle> NeonWorkloadFactory::CreateSubTensorHandle(ITensorHandle& parent,
    TensorShape const& subTensorShape,
    unsigned int const* subTensorOrigin) const
{
    return nullptr;
}

std::unique_ptr<ITensorHandle> NeonWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                            const WorkloadInfo&        info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateOutput(const OutputQueueDescriptor& descriptor,
                                                             const WorkloadInfo&        info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateActivation(const ActivationQueueDescriptor& descriptor,
                                                                 const WorkloadInfo&              info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateSoftmax(const SoftmaxQueueDescriptor& descriptor,
                                                              const WorkloadInfo&           info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateSplitter(const SplitterQueueDescriptor& descriptor,
                                                               const WorkloadInfo&            info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateMerger(const MergerQueueDescriptor& descriptor,
                                                             const WorkloadInfo&          info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateFullyConnected(const FullyConnectedQueueDescriptor& descriptor,
                                                                     const WorkloadInfo&                  info) const
{
    return nullptr;
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreatePermute(const PermuteQueueDescriptor& descriptor,
                                                                     const WorkloadInfo&           info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreatePooling2d(const Pooling2dQueueDescriptor& descriptor,
                                                                const WorkloadInfo&           info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateConvolution2d(const Convolution2dQueueDescriptor& descriptor,
                                                                    const WorkloadInfo&               info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateDepthwiseConvolution2d(
    const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateNormalization(const NormalizationQueueDescriptor& descriptor,
                                                                    const WorkloadInfo&                 info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                               const WorkloadInfo&            info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateBatchNormalization(const BatchNormalizationQueueDescriptor& data,
                                                                         const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateMultiplication(const MultiplicationQueueDescriptor& data,
                                                                     const WorkloadInfo&                  info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                                              const WorkloadInfo&        info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateResizeBilinear(const ResizeBilinearQueueDescriptor& descriptor,
                                                                     const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateFakeQuantization(
        const FakeQuantizationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateL2Normalization(const L2NormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateConstant(const ConstantQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateReshape(const ReshapeQueueDescriptor& descriptor,
    const WorkloadInfo&           info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateFloor(const FloorQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

void NeonWorkloadFactory::Finalize()
{}

#endif

} //namespace armnn
