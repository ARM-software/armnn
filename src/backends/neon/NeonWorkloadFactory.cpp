//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonBackendId.hpp"
#include "NeonBackendModelContext.hpp"
#include "NeonTensorHandle.hpp"
#include "NeonWorkloadFactory.hpp"

#include <Layer.hpp>

#include <armnn/Utils.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/MakeWorkloadHelper.hpp>
#include <backendsCommon/MemCopyWorkload.hpp>
#include <backendsCommon/MemImportWorkload.hpp>

#include <neon/workloads/NeonWorkloadUtils.hpp>
#include <neon/workloads/NeonWorkloads.hpp>

namespace armnn
{

namespace
{
static const BackendId s_Id{NeonBackendId()};
}

bool NeonWorkloadFactory::IsLayerSupported(const Layer& layer,
                                           Optional<DataType> dataType,
                                           std::string& outReasonIfUnsupported)
{
    return IWorkloadFactory::IsLayerSupported(s_Id, layer, dataType, outReasonIfUnsupported);
}

bool NeonWorkloadFactory::IsLayerSupported(const IConnectableLayer& layer,
                                           Optional<DataType> dataType,
                                           std::string& outReasonIfUnsupported,
                                           const ModelOptions& modelOptions)
{
    return IWorkloadFactory::IsLayerSupported(s_Id, layer, dataType, outReasonIfUnsupported, modelOptions);
}

const BackendId& NeonWorkloadFactory::GetBackendId() const
{
    return s_Id;
}

NeonWorkloadFactory::NeonWorkloadFactory(const std::shared_ptr<NeonMemoryManager>& memoryManager)
    : m_MemoryManager(memoryManager), m_ModelContextPtr(IBackendInternal::IBackendSpecificModelContextPtr{})
{
}

NeonWorkloadFactory::NeonWorkloadFactory(const std::shared_ptr<NeonMemoryManager>& memoryManager,
                                         const IBackendInternal::IBackendSpecificModelContextPtr& modelContextPtr)
    : m_MemoryManager(memoryManager), m_ModelContextPtr(modelContextPtr)
{
}

std::unique_ptr<ITensorHandle> NeonWorkloadFactory::CreateSubTensorHandle(ITensorHandle& parent,
    TensorShape const& subTensorShape,
    unsigned int const* subTensorOrigin) const
{
    const arm_compute::TensorShape shape = armcomputetensorutils::BuildArmComputeTensorShape(subTensorShape);

    arm_compute::Coordinates coords;
    coords.set_num_dimensions(subTensorShape.GetNumDimensions());
    for (unsigned int i = 0; i < subTensorShape.GetNumDimensions(); i++)
    {
        // Arm compute indexes tensor coords in reverse order.
        unsigned int revertedIndex = subTensorShape.GetNumDimensions() - i - 1;
        coords.set(i, armnn::numeric_cast<int>(subTensorOrigin[revertedIndex]));
    }

    const arm_compute::TensorShape parentShape = armcomputetensorutils::BuildArmComputeTensorShape(parent.GetShape());
    if (!::arm_compute::error_on_invalid_subtensor(__func__, __FILE__, __LINE__, parentShape, coords, shape))
    {
        return nullptr;
    }

    return std::make_unique<NeonSubTensorHandle>(
        PolymorphicDowncast<IAclTensorHandle*>(&parent), shape, coords);
}

std::unique_ptr<ITensorHandle> NeonWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                       const bool IsMemoryManaged) const
{
    auto tensorHandle = std::make_unique<NeonTensorHandle>(tensorInfo);
    if (IsMemoryManaged)
    {
        tensorHandle->SetMemoryGroup(m_MemoryManager->GetInterLayerMemoryGroup());
    }
    return tensorHandle;
}

std::unique_ptr<ITensorHandle> NeonWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                       DataLayout dataLayout,
                                                                       const bool IsMemoryManaged) const
{
    auto tensorHandle = std::make_unique<NeonTensorHandle>(tensorInfo, dataLayout);
    if (IsMemoryManaged)
    {
        tensorHandle->SetMemoryGroup(m_MemoryManager->GetInterLayerMemoryGroup());
    }
    return tensorHandle;
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateAbs(const AbsQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    IgnoreUnused(descriptor);

    ElementwiseUnaryQueueDescriptor elementwiseUnaryDescriptor;
    elementwiseUnaryDescriptor.m_Parameters = ElementwiseUnaryDescriptor(UnaryOperation::Abs);

    return CreateElementwiseUnary(elementwiseUnaryDescriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateActivation(const ActivationQueueDescriptor& descriptor,
                                                                 const WorkloadInfo&              info) const
{
    return std::make_unique<NeonActivationWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                                      const WorkloadInfo&            info) const
{
    return std::make_unique<NeonAdditionWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateArgMinMax(const ArgMinMaxQueueDescriptor& descriptor,
                                                                const WorkloadInfo& info) const
{
    return std::make_unique<NeonArgMinMaxWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateBatchNormalization(
    const BatchNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return std::make_unique<NeonBatchNormalizationWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateBatchToSpaceNd(const BatchToSpaceNdQueueDescriptor& descriptor,
                                                                     const WorkloadInfo& info) const
{
    return std::make_unique<NeonBatchToSpaceNdWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateComparison(const ComparisonQueueDescriptor& descriptor,
                                                                 const WorkloadInfo& info) const
{
    return std::make_unique<NeonComparisonWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateConcat(const ConcatQueueDescriptor& descriptor,
                                                                    const WorkloadInfo&          info) const
{
    return std::make_unique<NeonConcatWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateConstant(const ConstantQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    return std::make_unique<NeonConstantWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateConvertBf16ToFp32(
    const ConvertBf16ToFp32QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<NeonConvertBf16ToFp32Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateConvertFp16ToFp32(
    const ConvertFp16ToFp32QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<NeonConvertFp16ToFp32Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateConvertFp32ToBf16(
    const ConvertFp32ToBf16QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<NeonConvertFp32ToBf16Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateConvertFp32ToFp16(
    const ConvertFp32ToFp16QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<NeonConvertFp32ToFp16Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateConvolution2d(
    const Convolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    bool isFastMathEnabled = false;
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
    return std::make_unique<NeonConvolution2dWorkload>(descriptor,
                                                       info,
                                                       m_MemoryManager->GetIntraLayerManager(),
                                                       isFastMathEnabled);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateDebug(const DebugQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return MakeWorkloadHelper<NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateDepthToSpace(const DepthToSpaceQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const
{
    return std::make_unique<NeonDepthToSpaceWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateDepthwiseConvolution2d(
    const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return std::make_unique<NeonDepthwiseConvolutionWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateDequantize(const DequantizeQueueDescriptor& descriptor,
                                                                 const WorkloadInfo& info) const
{
    return std::make_unique<NeonDequantizeWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateDetectionPostProcess(
    const armnn::DetectionPostProcessQueueDescriptor& descriptor, const armnn::WorkloadInfo& info) const
{
    return MakeWorkloadHelper<NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateDivision(
    const DivisionQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return std::make_unique<NeonDivisionWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateElementwiseUnary(
    const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    switch(descriptor.m_Parameters.m_Operation)
    {
        case UnaryOperation::Abs:
        {
            AbsQueueDescriptor absQueueDescriptor;
            absQueueDescriptor.m_Inputs  = descriptor.m_Inputs;
            absQueueDescriptor.m_Outputs = descriptor.m_Outputs;

            return std::make_unique<NeonAbsWorkload>(absQueueDescriptor, info);
        }
        case UnaryOperation::Rsqrt:
        {
            RsqrtQueueDescriptor rsqrtQueueDescriptor;
            rsqrtQueueDescriptor.m_Inputs  = descriptor.m_Inputs;
            rsqrtQueueDescriptor.m_Outputs = descriptor.m_Outputs;

            return std::make_unique<NeonRsqrtWorkload>(rsqrtQueueDescriptor, info);
        }
        case UnaryOperation::Neg:
            return std::make_unique<NeonNegWorkload>(descriptor, info);
        case UnaryOperation::Exp:
            return std::make_unique<NeonExpWorkload>(descriptor, info);
        case UnaryOperation::LogicalNot:
            return std::make_unique<NeonLogicalNotWorkload>(descriptor, info);
        default:
            return nullptr;
    }
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateEqual(const EqualQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    IgnoreUnused(descriptor);

    ComparisonQueueDescriptor comparisonDescriptor;
    comparisonDescriptor.m_Parameters = ComparisonDescriptor(ComparisonOperation::Equal);

    return CreateComparison(comparisonDescriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateFill(const FillQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return std::make_unique<NeonFillWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateFloor(const FloorQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return MakeWorkloadHelper<NeonFloorFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateFullyConnected(
    const FullyConnectedQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return std::make_unique<NeonFullyConnectedWorkload>(descriptor, info, m_MemoryManager->GetIntraLayerManager());
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateGather(const armnn::GatherQueueDescriptor& descriptor,
                                                             const armnn::WorkloadInfo& info) const
{
    return std::make_unique<NeonGatherWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateGreater(const GreaterQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    IgnoreUnused(descriptor);

    ComparisonQueueDescriptor comparisonDescriptor;
    comparisonDescriptor.m_Parameters = ComparisonDescriptor(ComparisonOperation::Greater);

    return CreateComparison(comparisonDescriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                            const WorkloadInfo&        info) const
{
    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateInstanceNormalization(
    const InstanceNormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return std::make_unique<NeonInstanceNormalizationWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateL2Normalization(const L2NormalizationQueueDescriptor& descriptor,
                                                                      const WorkloadInfo& info) const
{
    return MakeWorkloadHelper<NeonL2NormalizationFloatWorkload, NullWorkload>(descriptor, info,
                                                                              m_MemoryManager->GetIntraLayerManager());
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateLogSoftmax(const LogSoftmaxQueueDescriptor& descriptor,
                                                                 const WorkloadInfo& info) const
{
    return std::make_unique<NeonLogSoftmaxWorkload>(descriptor, info, m_MemoryManager->GetIntraLayerManager());
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateLogicalBinary(const LogicalBinaryQueueDescriptor& descriptor,
                                                                    const WorkloadInfo& info) const
{
    switch(descriptor.m_Parameters.m_Operation)
    {
        case LogicalBinaryOperation::LogicalAnd:
            return std::make_unique<NeonLogicalAndWorkload>(descriptor, info);
        case LogicalBinaryOperation::LogicalOr:
            return std::make_unique<NeonLogicalOrWorkload>(descriptor, info);
        default:
            return nullptr;
    }
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateLstm(const LstmQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return MakeWorkloadHelper<NeonLstmFloatWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateMaximum(const MaximumQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    return std::make_unique<NeonMaximumWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateMean(const MeanQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return std::make_unique<NeonMeanWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                                                     const WorkloadInfo&        info) const
{
    if (descriptor.m_Inputs.empty() || !descriptor.m_Inputs[0])
    {
        throw InvalidArgumentException("NeonWorkloadFactory: Invalid null input for MemCopy workload");
    }

    return MakeWorkloadHelper<CopyMemGenericWorkload, CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateMemImport(const MemImportQueueDescriptor& descriptor,
                                                                       const WorkloadInfo&        info) const
{
    if (descriptor.m_Inputs.empty() || !descriptor.m_Inputs[0])
    {
        throw InvalidArgumentException("NeonWorkloadFactory: Invalid null input for MemImport workload");
    }

    return std::make_unique<ImportMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateMerger(const MergerQueueDescriptor& descriptor,
                                                                    const WorkloadInfo&          info) const
{
    return CreateConcat(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateMinimum(const MinimumQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    return std::make_unique<NeonMinimumWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateMultiplication(
    const MultiplicationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return std::make_unique<NeonMultiplicationWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateNormalization(
    const NormalizationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkloadHelper<NeonNormalizationFloatWorkload, NullWorkload>(descriptor, info,
                                                                            m_MemoryManager->GetIntraLayerManager());
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateOutput(const OutputQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreatePad(const PadQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return std::make_unique<NeonPadWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreatePermute(const PermuteQueueDescriptor& descriptor,
                                                                     const WorkloadInfo& info) const
{
    return std::make_unique<NeonPermuteWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreatePooling2d(const Pooling2dQueueDescriptor& descriptor,
                                                                       const WorkloadInfo& info) const
{
    return std::make_unique<NeonPooling2dWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreatePreCompiled(const PreCompiledQueueDescriptor& descriptor,
                                                                  const WorkloadInfo& info) const
{
    return MakeWorkloadHelper<NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreatePrelu(const armnn::PreluQueueDescriptor &descriptor,
                                                                   const armnn::WorkloadInfo &info) const
{
    return std::make_unique<NeonPreluWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateQLstm(const QLstmQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return std::make_unique<NeonQLstmWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateQuantize(const QuantizeQueueDescriptor& descriptor,
                                                                      const WorkloadInfo& info) const
{
    return std::make_unique<NeonQuantizeWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateQuantizedLstm(const QuantizedLstmQueueDescriptor& descriptor,
                                                                    const WorkloadInfo& info) const
{
    return std::make_unique<NeonQuantizedLstmWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateReshape(const ReshapeQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    return std::make_unique<NeonReshapeWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateResize(const ResizeQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    return std::make_unique<NeonResizeWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateResizeBilinear(
    const ResizeBilinearQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    ResizeQueueDescriptor resizeDescriptor;
    resizeDescriptor.m_Inputs  = descriptor.m_Inputs;
    resizeDescriptor.m_Outputs = descriptor.m_Outputs;

    resizeDescriptor.m_Parameters.m_DataLayout   = descriptor.m_Parameters.m_DataLayout;
    resizeDescriptor.m_Parameters.m_TargetWidth  = descriptor.m_Parameters.m_TargetWidth;
    resizeDescriptor.m_Parameters.m_TargetHeight = descriptor.m_Parameters.m_TargetHeight;

    return CreateResize(resizeDescriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateRsqrt(const RsqrtQueueDescriptor &descriptor,
                                                            const WorkloadInfo &info) const
{
    IgnoreUnused(descriptor);

    ElementwiseUnaryQueueDescriptor elementwiseUnaryDescriptor;
    elementwiseUnaryDescriptor.m_Parameters = ElementwiseUnaryDescriptor(UnaryOperation::Rsqrt);

    return CreateElementwiseUnary(elementwiseUnaryDescriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateSlice(const SliceQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return std::make_unique<NeonSliceWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateSoftmax(const SoftmaxQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    return std::make_unique<NeonSoftmaxWorkload>(descriptor, info, m_MemoryManager->GetIntraLayerManager());
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateSpaceToBatchNd(const SpaceToBatchNdQueueDescriptor& descriptor,
                                                                     const WorkloadInfo& info) const
{
    return std::make_unique<NeonSpaceToBatchNdWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateSpaceToDepth(const SpaceToDepthQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const
{
    return std::make_unique<NeonSpaceToDepthWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateSplitter(const SplitterQueueDescriptor& descriptor,
                                                               const WorkloadInfo&            info) const
{
    return std::make_unique<NeonSplitterWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateStack(const StackQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return std::make_unique<NeonStackWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateStridedSlice(const StridedSliceQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const
{
    return std::make_unique<NeonStridedSliceWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateSubtraction(
    const SubtractionQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return std::make_unique<NeonSubtractionWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NeonWorkloadFactory::CreateTranspose(const TransposeQueueDescriptor& descriptor,
                                                                       const WorkloadInfo& info) const
{
    return std::make_unique<NeonTransposeWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateTransposeConvolution2d(
    const TransposeConvolution2dQueueDescriptor &descriptor,
    const WorkloadInfo &info) const
{
    return std::make_unique<NeonTransposeConvolution2dWorkload>(descriptor, info,
                                                                m_MemoryManager->GetIntraLayerManager());
}

} // namespace armnn
