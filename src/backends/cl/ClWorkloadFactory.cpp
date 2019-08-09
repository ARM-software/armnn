//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ClWorkloadFactory.hpp"
#include "ClBackendId.hpp"

#include <Layer.hpp>

#include <armnn/Exceptions.hpp>
#include <armnn/Utils.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/MakeWorkloadHelper.hpp>
#include <backendsCommon/MemCopyWorkload.hpp>
#include <backendsCommon/MemImportWorkload.hpp>

#include <cl/ClTensorHandle.hpp>
#include <cl/workloads/ClWorkloads.hpp>
#include <cl/workloads/ClWorkloadUtils.hpp>

#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/runtime/CL/CLBufferAllocator.h>
#include <arm_compute/runtime/CL/CLScheduler.h>

#include <boost/polymorphic_cast.hpp>
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>

namespace armnn
{

namespace
{
static const BackendId s_Id{ClBackendId()};
}

bool ClWorkloadFactory::IsLayerSupported(const Layer& layer,
                                         Optional<DataType> dataType,
                                         std::string& outReasonIfUnsupported)
{
    return IWorkloadFactory::IsLayerSupported(s_Id, layer, dataType, outReasonIfUnsupported);
}

const BackendId& ClWorkloadFactory::GetBackendId() const
{
    return s_Id;
}

template <typename FloatWorkload, typename Uint8Workload, typename QueueDescriptorType, typename... Args>
std::unique_ptr<IWorkload> ClWorkloadFactory::MakeWorkload(const QueueDescriptorType& descriptor,
                                                           const WorkloadInfo& info,
                                                           Args&&... args)
{
    try
    {
        return MakeWorkloadHelper<FloatWorkload, Uint8Workload>(descriptor, info, std::forward<Args>(args)...);
    }
    catch (const cl::Error& clError)
    {
        throw WrapClError(clError, CHECK_LOCATION());
    }
}

template <typename Workload, typename QueueDescriptorType, typename... Args>
std::unique_ptr<IWorkload> ClWorkloadFactory::MakeWorkload(const QueueDescriptorType& descriptor,
                                                           const WorkloadInfo& info,
                                                           Args&&... args)
{
    try
    {
        return std::make_unique<Workload>(descriptor, info, std::forward<Args>(args)...);
    }
    catch (const cl::Error& clError)
    {
        throw WrapClError(clError, CHECK_LOCATION());
    }
}

ClWorkloadFactory::ClWorkloadFactory(const std::shared_ptr<ClMemoryManager>& memoryManager)
    : m_MemoryManager(memoryManager)
{
}

std::unique_ptr<ITensorHandle> ClWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    std::unique_ptr<ClTensorHandle> tensorHandle = std::make_unique<ClTensorHandle>(tensorInfo);
    tensorHandle->SetMemoryGroup(m_MemoryManager->GetInterLayerMemoryGroup());

    return tensorHandle;
}

std::unique_ptr<ITensorHandle> ClWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                     DataLayout dataLayout) const
{
    std::unique_ptr<ClTensorHandle> tensorHandle = std::make_unique<ClTensorHandle>(tensorInfo, dataLayout);
    tensorHandle->SetMemoryGroup(m_MemoryManager->GetInterLayerMemoryGroup());

    return tensorHandle;
}

std::unique_ptr<ITensorHandle> ClWorkloadFactory::CreateSubTensorHandle(ITensorHandle&      parent,
                                                                        TensorShape const&   subTensorShape,
                                                                        unsigned int const* subTensorOrigin) const
{
    arm_compute::Coordinates coords;
    arm_compute::TensorShape shape = armcomputetensorutils::BuildArmComputeTensorShape(subTensorShape);

    coords.set_num_dimensions(subTensorShape.GetNumDimensions());
    for (unsigned int i = 0; i < subTensorShape.GetNumDimensions(); i++)
    {
        // Arm compute indexes tensor coords in reverse order.
        unsigned int revertedIndex = subTensorShape.GetNumDimensions() - i - 1;
        coords.set(i, boost::numeric_cast<int>(subTensorOrigin[revertedIndex]));
    }

    const arm_compute::TensorShape parentShape = armcomputetensorutils::BuildArmComputeTensorShape(parent.GetShape());
    if (!::arm_compute::error_on_invalid_subtensor(__func__, __FILE__, __LINE__, parentShape, coords, shape))
    {
        return nullptr;
    }

    return std::make_unique<ClSubTensorHandle>(
        boost::polymorphic_downcast<IClTensorHandle*>(&parent), shape, coords);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateOutput(const OutputQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateActivation(const ActivationQueueDescriptor& descriptor,
                                                               const WorkloadInfo&              info) const
{
    return MakeWorkload<ClActivationWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSoftmax(const SoftmaxQueueDescriptor& descriptor,
                                                            const WorkloadInfo&           info) const
{
    return MakeWorkload<ClSoftmaxFloatWorkload, ClSoftmaxUint8Workload>(descriptor, info,
                                                                        m_MemoryManager->GetIntraLayerManager());
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSplitter(const SplitterQueueDescriptor& descriptor,
                                                             const WorkloadInfo&            info) const
{
    return MakeWorkload<ClSplitterWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateMerger(const MergerQueueDescriptor& descriptor,
                                                                  const WorkloadInfo&          info) const
{
    return CreateConcat(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateFullyConnected(
    const FullyConnectedQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<ClFullyConnectedWorkload, ClFullyConnectedWorkload>(descriptor, info,
                                                                            m_MemoryManager->GetIntraLayerManager());
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreatePermute(const PermuteQueueDescriptor& descriptor,
                                                                   const WorkloadInfo&           info) const
{
    return MakeWorkload<ClPermuteWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreatePooling2d(const Pooling2dQueueDescriptor& descriptor,
                                                                     const WorkloadInfo&           info) const
{
    return MakeWorkload<ClPooling2dWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreatePrelu(const armnn::PreluQueueDescriptor &descriptor,
                                                                 const armnn::WorkloadInfo &info) const
{
    return MakeWorkload<ClPreluWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateConvolution2d(const Convolution2dQueueDescriptor& descriptor,
                                                                         const WorkloadInfo&               info) const
{
    return MakeWorkload<ClConvolution2dWorkload>(descriptor, info, m_MemoryManager->GetIntraLayerManager());
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateDepthwiseConvolution2d(
    const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<ClDepthwiseConvolutionWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateDetectionPostProcess(
    const armnn::DetectionPostProcessQueueDescriptor& descriptor, const armnn::WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateDequantize(const DequantizeQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    return std::make_unique<ClDequantizeWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateNormalization(const NormalizationQueueDescriptor& descriptor,
                                                                         const WorkloadInfo&                 info) const
{
    return MakeWorkload<ClNormalizationFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                                    const WorkloadInfo&            info) const
{
    return MakeWorkload<ClAdditionWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateMultiplication(
    const MultiplicationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<ClMultiplicationWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateDivision(
    const DivisionQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<ClDivisionFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateSubtraction(const SubtractionQueueDescriptor& descriptor,
                                                                       const WorkloadInfo& info) const
{
    return MakeWorkload<ClSubtractionWorkload>(descriptor, info);
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

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateMemImport(const MemImportQueueDescriptor& descriptor,
                                                                     const WorkloadInfo& info) const
{
    if (descriptor.m_Inputs.empty() || !descriptor.m_Inputs[0])
    {
        throw InvalidArgumentException("ClWorkloadFactory: Invalid null input for MemImport workload");
    }

    return std::make_unique<ImportMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateResize(const ResizeQueueDescriptor& descriptor,
                                                                  const WorkloadInfo& info) const
{
    return MakeWorkload<ClResizeWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateResizeBilinear(
    const ResizeBilinearQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    ResizeQueueDescriptor resizeDescriptor;
    resizeDescriptor.m_Inputs  = descriptor.m_Inputs;
    resizeDescriptor.m_Outputs = descriptor.m_Outputs;

    resizeDescriptor.m_Parameters.m_Method       = ResizeMethod::Bilinear;
    resizeDescriptor.m_Parameters.m_DataLayout   = descriptor.m_Parameters.m_DataLayout;
    resizeDescriptor.m_Parameters.m_TargetHeight = descriptor.m_Parameters.m_TargetHeight;
    resizeDescriptor.m_Parameters.m_TargetWidth  = descriptor.m_Parameters.m_TargetWidth;

    return CreateResize(resizeDescriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateFakeQuantization(
    const FakeQuantizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return nullptr;
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateQuantize(const QuantizeQueueDescriptor& descriptor,
                                                                    const WorkloadInfo& info) const
{
    return std::make_unique<ClQuantizeWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateL2Normalization(const L2NormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClL2NormalizationFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateConcat(const ConcatQueueDescriptor& descriptor,
                                                                  const WorkloadInfo&          info) const
{
    return MakeWorkload<ClConcatWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConstant(const ConstantQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClConstantWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateReshape(const ReshapeQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClReshapeWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSpaceToBatchNd(const SpaceToBatchNdQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClSpaceToBatchNdWorkload>(descriptor, info);
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
    return MakeWorkload<ClConvertFp16ToFp32Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConvertFp32ToFp16(
    const ConvertFp32ToFp16QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClConvertFp32ToFp16Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateMaximum(const MaximumQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return MakeWorkload<ClMaximumWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateMean(const MeanQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info) const
{
    return std::make_unique<ClMeanWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreatePad(const PadQueueDescriptor& descriptor,
                                                        const WorkloadInfo& info) const
{
    return MakeWorkload<ClPadWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateEqual(const EqualQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateBatchToSpaceNd(const BatchToSpaceNdQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const
{
    return MakeWorkload<ClBatchToSpaceNdWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateStridedSlice(const StridedSliceQueueDescriptor& descriptor,
                                                                 const WorkloadInfo& info) const
{
    return MakeWorkload<ClStridedSliceWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateMinimum(const MinimumQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return MakeWorkload<ClMinimumWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateGreater(const GreaterQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return MakeWorkload<ClGreaterFloat32Workload, ClGreaterUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateDebug(const DebugQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateRsqrt(const RsqrtQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreatePreCompiled(const PreCompiledQueueDescriptor& descriptor,
                                                                const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateGather(const armnn::GatherQueueDescriptor& descriptor,
                                                           const armnn::WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> ClWorkloadFactory::CreateTransposeConvolution2d(
    const TransposeConvolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClTransposeConvolution2dWorkload>(descriptor, info, m_MemoryManager->GetIntraLayerManager());
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSpaceToDepth(const SpaceToDepthQueueDescriptor& descriptor,
                                                                 const WorkloadInfo& info) const
{
    return MakeWorkload<ClSpaceToDepthWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateQuantizedLstm(const QuantizedLstmQueueDescriptor& descriptor,
                                                                  const WorkloadInfo& info) const
{
    return MakeWorkload<ClQuantizedLstmWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateStack(const StackQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return MakeWorkload<ClStackWorkload>(descriptor, info);
}

} // namespace armnn
