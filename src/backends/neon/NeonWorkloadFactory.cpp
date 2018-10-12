//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "NeonWorkloadFactory.hpp"
#include <armnn/Utils.hpp>
#include <backends/CpuTensorHandle.hpp>
#include <Layer.hpp>

#ifdef ARMCOMPUTENEON_ENABLED
#include <arm_compute/runtime/Allocator.h>

#include <backends/MemCopyWorkload.hpp>
#include "NeonTensorHandle.hpp"
#include "workloads/NeonWorkloadUtils.hpp"
#include "workloads/NeonWorkloads.hpp"

#include <memory/IPoolManager.hpp>
#endif

#include <backends/MakeWorkloadHelper.hpp>

#include <boost/polymorphic_cast.hpp>

namespace armnn
{

bool NeonWorkloadFactory::IsLayerSupported(const Layer& layer, boost::optional<DataType> dataType,
                                           std::string& outReasonIfUnsupported)
{
    return IWorkloadFactory::IsLayerSupported(Compute::CpuAcc, layer, dataType, outReasonIfUnsupported);
}

#ifdef ARMCOMPUTENEON_ENABLED

NeonWorkloadFactory::NeonWorkloadFactory()
    : m_MemoryManager(std::make_unique<arm_compute::Allocator>(), BaseMemoryManager::MemoryAffinity::Offset)
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
        // Arm compute indexes tensor coords in reverse order.
        unsigned int revertedIndex = subTensorShape.GetNumDimensions() - i - 1;
        coords.set(i, boost::numeric_cast<int>(subTensorOrigin[revertedIndex]));
    }

    return std::make_unique<NeonSubTensorHandle>(
        boost::polymorphic_downcast<INeonTensorHandle*>(&parent), shape, coords);
}

std::unique_ptr<ITensorHandle> NeonWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    auto tensorHandle = std::make_unique<NeonTensorHandle>(tensorInfo);
    tensorHandle->SetMemoryGroup(m_MemoryManager.GetInterLayerMemoryGroup());

    return tensorHandle;
}

std::unique_ptr<ITensorHandle> NeonWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                       DataLayout dataLayout) const
{
    auto tensorHandle = std::make_unique<NeonTensorHandle>(tensorInfo, dataLayout);
    tensorHandle->SetMemoryGroup(m_MemoryManager.GetInterLayerMemoryGroup());

    return tensorHandle;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                            const WorkloadInfo&        info) const
{
    return MakeWorkload<CopyMemGenericWorkload, CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateOutput(const OutputQueueDescriptor& descriptor,
                                                             const WorkloadInfo&        info) const
{
    return MakeWorkload<CopyMemGenericWorkload, CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateActivation(const ActivationQueueDescriptor& descriptor,
                                                                 const WorkloadInfo&              info) const
{
    return std::make_unique<NeonActivationWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateSoftmax(const SoftmaxQueueDescriptor& descriptor,
                                                              const WorkloadInfo&           info) const
{
    return MakeWorkload<NeonSoftmaxFloatWorkload, NeonSoftmaxUint8Workload>(descriptor, info,
                                                                              m_MemoryManager.GetIntraLayerManager());
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateSplitter(const SplitterQueueDescriptor& descriptor,
                                                               const WorkloadInfo&            info) const
{
    return MakeWorkload<NeonSplitterFloatWorkload, NeonSplitterUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateMerger(const MergerQueueDescriptor& descriptor,
                                                                    const WorkloadInfo&          info) const
{
    return std::make_unique<NeonMergerWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateFullyConnected(
    const FullyConnectedQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NeonFullyConnectedWorkload, NeonFullyConnectedWorkload>(descriptor, info,
                                                                                m_MemoryManager.GetIntraLayerManager());
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreatePermute(const PermuteQueueDescriptor& descriptor,
                                                                     const WorkloadInfo&           info) const
{
    return MakeWorkload<NeonPermuteFloatWorkload, NeonPermuteUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreatePooling2d(const Pooling2dQueueDescriptor& descriptor,
                                                                       const WorkloadInfo&           info) const
{
    return MakeWorkload<NeonPooling2dFloatWorkload, NeonPooling2dUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateConvolution2d(
    const Convolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NeonConvolution2dFloatWorkload, NeonConvolution2dUint8Workload>(descriptor, info,
                                                                              m_MemoryManager.GetIntraLayerManager());
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateDepthwiseConvolution2d(
    const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NeonDepthwiseConvolutionFloatWorkload, NeonDepthwiseConvolutionUint8Workload>(
        descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateNormalization(
    const NormalizationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NeonNormalizationFloatWorkload, NullWorkload>(descriptor, info,
                                                                        m_MemoryManager.GetIntraLayerManager());
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                                      const WorkloadInfo&            info) const
{
    return MakeWorkload<NeonAdditionFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateMultiplication(
    const MultiplicationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NeonMultiplicationFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateDivision(
    const DivisionQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateSubtraction(
    const SubtractionQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NeonSubtractionFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateBatchNormalization(
    const BatchNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NeonBatchNormalizationFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                                                     const WorkloadInfo&        info) const
{
    if (descriptor.m_Inputs.empty() || !descriptor.m_Inputs[0])
    {
        throw InvalidArgumentException("NeonWorkloadFactory: Invalid null input for MemCopy workload");
    }

    return MakeWorkload<CopyMemGenericWorkload, CopyMemGenericWorkload>(descriptor, info);
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
    return MakeWorkload<NeonL2NormalizationFloatWorkload, NullWorkload>(descriptor, info,
                                                                          m_MemoryManager.GetIntraLayerManager());
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateConstant(const ConstantQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<NeonConstantFloatWorkload, NeonConstantUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateReshape(const ReshapeQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<NeonReshapeFloatWorkload, NeonReshapeUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateFloor(const FloorQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<NeonFloorFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateLstm(const LstmQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<NeonLstmFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateConvertFp16ToFp32(
    const ConvertFp16ToFp32QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<NeonConvertFp16ToFp32Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateConvertFp32ToFp16(
    const ConvertFp32ToFp16QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<NeonConvertFp32ToFp16Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateMean(const MeanQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreatePad(const PadQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
}

void NeonWorkloadFactory::Finalize()
{
    m_MemoryManager.Finalize();
}

void NeonWorkloadFactory::Release()
{
    m_MemoryManager.Release();
}

void NeonWorkloadFactory::Acquire()
{
    m_MemoryManager.Acquire();
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

std::unique_ptr<ITensorHandle> NeonWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                       DataLayout dataLayout) const
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

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateLstm(const LstmQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateConvertFp16ToFp32(
    const ConvertFp16ToFp32QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateConvertFp32ToFp16(
    const ConvertFp32ToFp16QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateDivision(const DivisionQueueDescriptor& data,
                                                               const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateSubtraction(const SubtractionQueueDescriptor& data,
                                                                  const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateMean(const MeanQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreatePad(const PadQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return nullptr;
}

void NeonWorkloadFactory::Finalize()
{}

void NeonWorkloadFactory::Release()
{}

void NeonWorkloadFactory::Acquire()
{}

#endif

} //namespace armnn
