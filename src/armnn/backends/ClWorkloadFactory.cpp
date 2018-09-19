//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ClWorkloadFactory.hpp"

#include "armnn/Exceptions.hpp"
#include "armnn/Utils.hpp"

#include <string>
#include "CpuTensorHandle.hpp"
#include "Layer.hpp"

#ifdef ARMCOMPUTECL_ENABLED
#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/runtime/CL/CLBufferAllocator.h>
#include <arm_compute/runtime/CL/CLScheduler.h>

#include "ClWorkloads.hpp"

#include "backends/MemCopyWorkload.hpp"
#include "backends/ClTensorHandle.hpp"

#include "memory/IPoolManager.hpp"
#endif

#include "MakeWorkloadHelper.hpp"

#include <boost/polymorphic_cast.hpp>
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>

namespace armnn
{

bool ClWorkloadFactory::IsLayerSupported(const Layer& layer,
                                         boost::optional<DataType> dataType,
                                         std::string& outReasonIfUnsupported)
{
    return IWorkloadFactory::IsLayerSupported(Compute::GpuAcc, layer, dataType, outReasonIfUnsupported);
}

#ifdef ARMCOMPUTECL_ENABLED

ClWorkloadFactory::ClWorkloadFactory()
: m_MemoryManager(std::make_unique<arm_compute::CLBufferAllocator>())
{
}

std::unique_ptr<ITensorHandle> ClWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    std::unique_ptr<ClTensorHandle> tensorHandle = std::make_unique<ClTensorHandle>(tensorInfo);
    tensorHandle->SetMemoryGroup(m_MemoryManager.GetInterLayerMemoryGroup());

    return tensorHandle;
}

std::unique_ptr<ITensorHandle> ClWorkloadFactory::CreateSubTensorHandle(ITensorHandle&      parent,
                                                                        TensorShape const&   subTensorShape,
                                                                        unsigned int const* subTensorOrigin) const
{
    BOOST_ASSERT(parent.GetType() == ITensorHandle::CL);

    arm_compute::Coordinates coords;
    arm_compute::TensorShape shape = armcomputetensorutils::BuildArmComputeTensorShape(subTensorShape);

    coords.set_num_dimensions(subTensorShape.GetNumDimensions());
    for (unsigned int i = 0; i < subTensorShape.GetNumDimensions(); i++)
    {
        // Arm compute indexes tensor coords in reverse order.
        unsigned int revertedIndex = subTensorShape.GetNumDimensions() - i - 1;
        coords.set(i, boost::numeric_cast<int>(subTensorOrigin[revertedIndex]));
    }

    return std::make_unique<ClSubTensorHandle>(
        boost::polymorphic_downcast<IClTensorHandle*>(&parent), shape, coords);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return MakeWorkload<CopyMemGenericWorkload, CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateOutput(const OutputQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return MakeWorkload<CopyMemGenericWorkload, CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateActivation(const ActivationQueueDescriptor& descriptor,
                                                               const WorkloadInfo&              info) const
{
    return MakeWorkload<ClActivationFloatWorkload, ClActivationUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSoftmax(const SoftmaxQueueDescriptor& descriptor,
                                                            const WorkloadInfo&           info) const
{
    return MakeWorkload<ClSoftmaxFloatWorkload, ClSoftmaxUint8Workload>(descriptor, info,
                                                                          m_MemoryManager.GetIntraLayerManager());
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSplitter(const SplitterQueueDescriptor& descriptor,
                                                             const WorkloadInfo&            info) const
{
    return MakeWorkload<ClSplitterFloatWorkload, ClSplitterUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateMerger(const MergerQueueDescriptor& descriptor,
                                                                  const WorkloadInfo&          info) const
{
    return MakeWorkload<ClMergerFloatWorkload, ClMergerUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateFullyConnected(
    const FullyConnectedQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<ClFullyConnectedFloatWorkload, NullWorkload>(descriptor, info,
                                                                       m_MemoryManager.GetIntraLayerManager());
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreatePermute(const PermuteQueueDescriptor& descriptor,
                                                                   const WorkloadInfo&           info) const
{
    return MakeWorkload<ClPermuteFloatWorkload, ClPermuteUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreatePooling2d(const Pooling2dQueueDescriptor& descriptor,
                                                                     const WorkloadInfo&           info) const
{
    return MakeWorkload<ClPooling2dFloatWorkload, ClPooling2dUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateConvolution2d(const Convolution2dQueueDescriptor& descriptor,
                                                                         const WorkloadInfo&               info) const
{
    return MakeWorkload<ClConvolution2dFloatWorkload, ClConvolution2dUint8Workload>(descriptor, info,
                                                                              m_MemoryManager.GetIntraLayerManager());
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateDepthwiseConvolution2d(
    const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<ClDepthwiseConvolutionFloatWorkload, ClDepthwiseConvolutionUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateNormalization(const NormalizationQueueDescriptor& descriptor,
                                                                         const WorkloadInfo&                 info) const
{
    return MakeWorkload<ClNormalizationFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                                    const WorkloadInfo&            info) const
{
    return MakeWorkload<ClAdditionWorkload<armnn::DataType::Float16, armnn::DataType::Float32>,
                        ClAdditionWorkload<armnn::DataType::QuantisedAsymm8>>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateMultiplication(
    const MultiplicationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<ClMultiplicationFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateDivision(
    const DivisionQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<ClDivisionFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateSubtraction(const SubtractionQueueDescriptor& descriptor,
                                                                       const WorkloadInfo& info) const
{
    return MakeWorkload<ClSubtractionWorkload<armnn::DataType::Float16, armnn::DataType::Float32>,
                        ClSubtractionWorkload<armnn::DataType::QuantisedAsymm8>>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateBatchNormalization(
    const BatchNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<ClBatchNormalizationFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const
{
    if (descriptor.m_Inputs.empty() || !descriptor.m_Inputs[0])
    {
        throw InvalidArgumentException("ClWorkloadFactory: Invalid null input for MemCopy workload");
    }

    return MakeWorkload<CopyMemGenericWorkload, CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateResizeBilinear(
    const ResizeBilinearQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClResizeBilinearFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateFakeQuantization(
    const FakeQuantizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateL2Normalization(const L2NormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClL2NormalizationFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConstant(const ConstantQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClConstantFloatWorkload, ClConstantUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateReshape(const ReshapeQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClReshapeFloatWorkload, ClReshapeUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateFloor(const FloorQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClFloorFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateLstm(const LstmQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClLstmFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConvertFp16ToFp32(
    const ConvertFp16ToFp32QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<ClConvertFp16ToFp32Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConvertFp32ToFp16(
    const ConvertFp32ToFp16QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<ClConvertFp32ToFp16Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateMean(const MeanQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreatePad(const PadQueueDescriptor& descriptor,
                                                        const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
}

void ClWorkloadFactory::Finalize()
{
    m_MemoryManager.Finalize();
}

void ClWorkloadFactory::Release()
{
    m_MemoryManager.Release();
}

void ClWorkloadFactory::Acquire()
{
    m_MemoryManager.Acquire();
}

#else // #if ARMCOMPUTECL_ENABLED

ClWorkloadFactory::ClWorkloadFactory()
{
}

std::unique_ptr<ITensorHandle> ClWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    return nullptr;
}

std::unique_ptr<ITensorHandle> ClWorkloadFactory::CreateSubTensorHandle(ITensorHandle&      parent,
                                                                        TensorShape const&   subTensorShape,
                                                                        unsigned int const* subTensorOrigin) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateOutput(const OutputQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateActivation(const ActivationQueueDescriptor& descriptor,
                                                               const WorkloadInfo&              info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSoftmax(const SoftmaxQueueDescriptor& descriptor,
                                                            const WorkloadInfo&           info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSplitter(const SplitterQueueDescriptor& descriptor,
                                                             const WorkloadInfo&            info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateMerger(const MergerQueueDescriptor& descriptor,
                                                           const WorkloadInfo&          info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateFullyConnected(const FullyConnectedQueueDescriptor& descriptor,
                                                                   const WorkloadInfo&                  info) const
{
    return nullptr;
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreatePermute(const PermuteQueueDescriptor& descriptor,
                                                                   const WorkloadInfo&           info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreatePooling2d(const Pooling2dQueueDescriptor& descriptor,
                                                              const WorkloadInfo&           info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConvolution2d(const Convolution2dQueueDescriptor& descriptor,
                                                                  const WorkloadInfo&               info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateDepthwiseConvolution2d(
    const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateNormalization(const NormalizationQueueDescriptor& descriptor,
                                                                  const WorkloadInfo&                 info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                             const WorkloadInfo&            info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateMultiplication(const MultiplicationQueueDescriptor& descriptor,
                                                                   const WorkloadInfo&                  info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateBatchNormalization(
    const BatchNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateResizeBilinear(const ResizeBilinearQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateFakeQuantization(const FakeQuantizationQueueDescriptor& descriptor,
                                                                     const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateL2Normalization(const L2NormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConstant(const ConstantQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateReshape(const ReshapeQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateFloor(const FloorQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateLstm(const LstmQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConvertFp16ToFp32(
    const ConvertFp16ToFp32QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConvertFp32ToFp16(
    const ConvertFp32ToFp16QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateDivision(const DivisionQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSubtraction(const SubtractionQueueDescriptor& descriptor,
                                                                const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateMean(const MeanQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreatePad(const PadQueueDescriptor& descriptor,
                                                        const WorkloadInfo& info) const
{
    return nullptr;
}

void ClWorkloadFactory::Finalize()
{
}

void ClWorkloadFactory::Release()
{
}

void ClWorkloadFactory::Acquire()
{
}

#endif // #if ARMCOMPUTECL_ENABLED

} // namespace armnn
