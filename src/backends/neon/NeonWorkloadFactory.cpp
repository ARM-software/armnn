//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
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

#include <backendsCommon/MakeWorkloadHelper.hpp>
#include <armnn/backends/MemCopyWorkload.hpp>
#include <backendsCommon/MemImportWorkload.hpp>
#include <armnn/backends/TensorHandle.hpp>

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

void NeonWorkloadFactory::SetNumberOfThreads()
{
    if (m_ModelContextPtr)
    {
        const unsigned int MIN_THREADS = 1;
        const unsigned int MAX_THREADS = 64;

        // Set the number of threads to be used if the user has set NumberOfThreads param
        // Only set if within limit or valid input
        auto modelOptions = dynamic_cast<NeonBackendModelContext*>(m_ModelContextPtr.get());
        auto numberOfThreads = modelOptions->GetNumberOfThreads();

        if (numberOfThreads != 0 && numberOfThreads >= MIN_THREADS && numberOfThreads <= MAX_THREADS)
        {
            arm_compute::Scheduler::get().set_num_threads(numberOfThreads);
        }
    }
}

NeonWorkloadFactory::NeonWorkloadFactory(const std::shared_ptr<NeonMemoryManager>& memoryManager)
    : m_MemoryManager(memoryManager), m_ModelContextPtr(IBackendInternal::IBackendSpecificModelContextPtr{})
{
    SetNumberOfThreads();
}

NeonWorkloadFactory::NeonWorkloadFactory(const std::shared_ptr<NeonMemoryManager>& memoryManager,
                                         const IBackendInternal::IBackendSpecificModelContextPtr& modelContextPtr)
    : m_MemoryManager(memoryManager), m_ModelContextPtr(modelContextPtr)
{
    SetNumberOfThreads();
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

std::unique_ptr<IWorkload> NeonWorkloadFactory::CreateWorkload(LayerType type,
                                                               const QueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    switch(type)
    {
        case LayerType::Activation :
        {
            auto activationQueueDescriptor = PolymorphicDowncast<const ActivationQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonActivationWorkload>(*activationQueueDescriptor, info);
        }
        case LayerType::Addition :
        {
            auto additionQueueDescriptor = PolymorphicDowncast<const AdditionQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonAdditionWorkload>(*additionQueueDescriptor, info);
        }
        case LayerType::ArgMinMax :
        {
            auto argMinMaxQueueDescriptor = PolymorphicDowncast<const ArgMinMaxQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonArgMinMaxWorkload>(*argMinMaxQueueDescriptor, info);
        }
        case LayerType::BatchMatMul :
        {
            auto batchMatMulQueueDescriptor = PolymorphicDowncast<const BatchMatMulQueueDescriptor*>(&descriptor);
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
            return std::make_unique<NeonBatchMatMulWorkload>(*batchMatMulQueueDescriptor, info, isFastMathEnabled);
        }
        case LayerType::BatchNormalization :
        {
            auto batchNormalizationQueueDescriptor
                     = PolymorphicDowncast<const BatchNormalizationQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonBatchNormalizationWorkload>(*batchNormalizationQueueDescriptor, info);
        }
        case LayerType::BatchToSpaceNd :
        {
            auto batchToSpaceNdQueueDescriptor
                     = PolymorphicDowncast<const BatchToSpaceNdQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonBatchToSpaceNdWorkload>(*batchToSpaceNdQueueDescriptor, info);
        }
        case LayerType::Cast :
        {
            auto castQueueDescriptor = PolymorphicDowncast<const CastQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonCastWorkload>(*castQueueDescriptor, info);
        }
        case LayerType::ChannelShuffle :
        {
            auto channelShuffleQueueDescriptor = PolymorphicDowncast<const ChannelShuffleQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonChannelShuffleWorkload>(*channelShuffleQueueDescriptor, info);
        }
        case LayerType::Comparison :
        {
            auto comparisonQueueDescriptor = PolymorphicDowncast<const ComparisonQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonComparisonWorkload>(*comparisonQueueDescriptor, info);
        }
        case LayerType::Concat :
        {
            auto concatQueueDescriptor = PolymorphicDowncast<const ConcatQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonConcatWorkload>(*concatQueueDescriptor, info);
        }
        case LayerType::Constant :
        {
            auto constantQueueDescriptor = PolymorphicDowncast<const ConstantQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonConstantWorkload>(*constantQueueDescriptor, info);
        }
        case LayerType::ConvertFp16ToFp32 :
        {
            auto convertFp16ToFp32QueueDescriptor
                     = PolymorphicDowncast<const ConvertFp16ToFp32QueueDescriptor*>(&descriptor);
            return std::make_unique<NeonConvertFp16ToFp32Workload>(*convertFp16ToFp32QueueDescriptor, info);
        }
        case LayerType::ConvertFp32ToFp16 :
        {
            auto convertFp32ToFp16QueueDescriptor
                     = PolymorphicDowncast<const ConvertFp32ToFp16QueueDescriptor*>(&descriptor);
            return std::make_unique<NeonConvertFp32ToFp16Workload>(*convertFp32ToFp16QueueDescriptor, info);
        }
        case LayerType::Convolution2d :
        {
            auto convolution2dQueueDescriptor = PolymorphicDowncast<const Convolution2dQueueDescriptor*>(&descriptor);
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
            return std::make_unique<NeonConvolution2dWorkload>(*convolution2dQueueDescriptor,
                                                               info,
                                                               m_MemoryManager->GetIntraLayerManager(),
                                                               isFastMathEnabled);
        }
        case LayerType::Convolution3d :
        {
            auto convolution3dQueueDescriptor = PolymorphicDowncast<const Convolution3dQueueDescriptor*>(&descriptor);
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
            return std::make_unique<NeonConvolution3dWorkload>(*convolution3dQueueDescriptor,
                                                               info,
                                                               m_MemoryManager->GetIntraLayerManager(),
                                                               isFastMathEnabled);
        }
        case LayerType::Debug :
        {
            auto debugQueueDescriptor = PolymorphicDowncast<const DebugQueueDescriptor*>(&descriptor);
            return MakeWorkloadHelper<NullWorkload, NullWorkload>(*debugQueueDescriptor, info);
        }
        case LayerType::DepthToSpace :
        {
            auto depthToSpaceQueueDescriptor = PolymorphicDowncast<const DepthToSpaceQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonDepthToSpaceWorkload>(*depthToSpaceQueueDescriptor, info);
        }
        case LayerType::DepthwiseConvolution2d :
        {
            auto depthwiseConvolution2dQueueDescriptor
                     = PolymorphicDowncast<const DepthwiseConvolution2dQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonDepthwiseConvolutionWorkload>(*depthwiseConvolution2dQueueDescriptor, info);
        }
        case LayerType::Dequantize :
        {
            auto dequantizeQueueDescriptor = PolymorphicDowncast<const DequantizeQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonDequantizeWorkload>(*dequantizeQueueDescriptor, info);
        }
        case LayerType::DetectionPostProcess :
        {
            auto detectionPostProcessQueueDescriptor
                     = PolymorphicDowncast<const DetectionPostProcessQueueDescriptor*>(&descriptor);
            return MakeWorkloadHelper<NullWorkload, NullWorkload>(*detectionPostProcessQueueDescriptor, info);
        }
        case LayerType::Division :
        {
            auto divisionQueueDescriptor = PolymorphicDowncast<const DivisionQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonDivisionWorkload>(*divisionQueueDescriptor, info);
        }
        case LayerType::ElementwiseBinary :
        {
            auto elementwiseBinaryQueueDescriptor
                     = PolymorphicDowncast<const ElementwiseBinaryQueueDescriptor*>(&descriptor);
            switch (elementwiseBinaryQueueDescriptor->m_Parameters.m_Operation)
            {
                case BinaryOperation::Add:
                {
                    AdditionQueueDescriptor additionQueueDescriptor;
                    additionQueueDescriptor.m_Inputs = descriptor.m_Inputs;
                    additionQueueDescriptor.m_Outputs = descriptor.m_Outputs;
                    return std::make_unique<NeonAdditionWorkload>(additionQueueDescriptor, info);
                }
                case BinaryOperation::Div:
                {
                    DivisionQueueDescriptor divisionQueueDescriptor;
                    divisionQueueDescriptor.m_Inputs = descriptor.m_Inputs;
                    divisionQueueDescriptor.m_Outputs = descriptor.m_Outputs;
                    return std::make_unique<NeonDivisionWorkload>(divisionQueueDescriptor, info);
                }
                case BinaryOperation::Maximum:
                {
                    MaximumQueueDescriptor maximumQueueDescriptor;
                    maximumQueueDescriptor.m_Inputs = descriptor.m_Inputs;
                    maximumQueueDescriptor.m_Outputs = descriptor.m_Outputs;
                    return std::make_unique<NeonMaximumWorkload>(maximumQueueDescriptor, info);
                }
                case BinaryOperation::Minimum:
                {
                    MinimumQueueDescriptor minimumQueueDescriptor;
                    minimumQueueDescriptor.m_Inputs = descriptor.m_Inputs;
                    minimumQueueDescriptor.m_Outputs = descriptor.m_Outputs;
                    return std::make_unique<NeonMinimumWorkload>(minimumQueueDescriptor, info);
                }
                case BinaryOperation::Mul:
                {
                    MultiplicationQueueDescriptor multiplicationQueueDescriptor;
                    multiplicationQueueDescriptor.m_Inputs = descriptor.m_Inputs;
                    multiplicationQueueDescriptor.m_Outputs = descriptor.m_Outputs;
                    return std::make_unique<NeonMultiplicationWorkload>(multiplicationQueueDescriptor, info);
                }
                case BinaryOperation::Power:
                case BinaryOperation::SqDiff:
                {
                    return std::make_unique<NeonElementwiseBinaryWorkload>(*elementwiseBinaryQueueDescriptor, info);
                }
                case BinaryOperation::Sub:
                {
                    SubtractionQueueDescriptor subtractionQueueDescriptor;
                    subtractionQueueDescriptor.m_Inputs = descriptor.m_Inputs;
                    subtractionQueueDescriptor.m_Outputs = descriptor.m_Outputs;
                    return std::make_unique<NeonSubtractionWorkload>(subtractionQueueDescriptor, info);
                }
                default:
                    return nullptr;
            }
        }
        case LayerType::ElementwiseUnary :
        {
            auto elementwiseUnaryQueueDescriptor
                     = PolymorphicDowncast<const ElementwiseUnaryQueueDescriptor*>(&descriptor);
            switch(elementwiseUnaryQueueDescriptor->m_Parameters.m_Operation)
            {
                case UnaryOperation::Abs:
                {
                    AbsQueueDescriptor absQueueDescriptor;
                    absQueueDescriptor.m_Inputs  = elementwiseUnaryQueueDescriptor->m_Inputs;
                    absQueueDescriptor.m_Outputs = elementwiseUnaryQueueDescriptor->m_Outputs;
                    return std::make_unique<NeonAbsWorkload>(absQueueDescriptor, info);
                }
                case UnaryOperation::Exp:
                    return std::make_unique<NeonExpWorkload>(*elementwiseUnaryQueueDescriptor, info);
                case UnaryOperation::LogicalNot:
                    return std::make_unique<NeonLogicalNotWorkload>(*elementwiseUnaryQueueDescriptor, info);
                case UnaryOperation::Log:
                    return std::make_unique<NeonLogWorkload>(*elementwiseUnaryQueueDescriptor, info);
                case UnaryOperation::Neg:
                    return std::make_unique<NeonNegWorkload>(*elementwiseUnaryQueueDescriptor, info);
                case UnaryOperation::Rsqrt:
                {
                    RsqrtQueueDescriptor rsqrtQueueDescriptor;
                    rsqrtQueueDescriptor.m_Inputs  = elementwiseUnaryQueueDescriptor->m_Inputs;
                    rsqrtQueueDescriptor.m_Outputs = elementwiseUnaryQueueDescriptor->m_Outputs;
                    return std::make_unique<NeonRsqrtWorkload>(rsqrtQueueDescriptor, info);
                }
                case UnaryOperation::Sin:
                    return std::make_unique<NeonSinWorkload>(*elementwiseUnaryQueueDescriptor, info);
                case UnaryOperation::Sqrt:
                    return std::make_unique<NeonSqrtWorkload>(*elementwiseUnaryQueueDescriptor, info);
                default:
                    return nullptr;
            }
        }
        case LayerType::Fill :
        {
            auto fillQueueDescriptor = PolymorphicDowncast<const FillQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonFillWorkload>(*fillQueueDescriptor, info);
        }
        case LayerType::Floor :
        {
            auto floorQueueDescriptor = PolymorphicDowncast<const FloorQueueDescriptor*>(&descriptor);
            return MakeWorkloadHelper<NeonFloorFloatWorkload, NullWorkload>(*floorQueueDescriptor, info);
        }
        case LayerType::FullyConnected :
        {
            auto fullyConnectedQueueDescriptor = PolymorphicDowncast<const FullyConnectedQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonFullyConnectedWorkload>(*fullyConnectedQueueDescriptor,
                                                                info,
                                                                m_MemoryManager->GetIntraLayerManager());
        }
        case LayerType::Fused :
        {
            auto fusedQueueDescriptor = PolymorphicDowncast<const FusedQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonFusedWorkload>(*fusedQueueDescriptor, info);
        }
        case LayerType::Gather :
        {
            auto gatherQueueDescriptor = PolymorphicDowncast<const GatherQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonGatherWorkload>(*gatherQueueDescriptor, info);
        }
        case LayerType::GatherNd :
        {
            auto gatherNdQueueDescriptor = PolymorphicDowncast<const GatherNdQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonGatherNdWorkload>(*gatherNdQueueDescriptor, info);
        }
        case LayerType::Input :
        {
            auto inputQueueDescriptor = PolymorphicDowncast<const InputQueueDescriptor*>(&descriptor);
            return std::make_unique<CopyMemGenericWorkload>(*inputQueueDescriptor, info);
        }
        case LayerType::InstanceNormalization :
        {
            auto instanceNormalizationQueueDescriptor
                     = PolymorphicDowncast<const InstanceNormalizationQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonInstanceNormalizationWorkload>(*instanceNormalizationQueueDescriptor, info);
        }
        case LayerType::L2Normalization :
        {
            auto l2NormalizationQueueDescriptor
                     = PolymorphicDowncast<const L2NormalizationQueueDescriptor*>(&descriptor);
            return MakeWorkloadHelper<NeonL2NormalizationFloatWorkload, NullWorkload>
                (*l2NormalizationQueueDescriptor, info, m_MemoryManager->GetIntraLayerManager());
        }
        case LayerType::LogSoftmax :
        {
            auto logSoftmaxQueueDescriptor = PolymorphicDowncast<const LogSoftmaxQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonLogSoftmaxWorkload>(*logSoftmaxQueueDescriptor,
                                                            info,
                                                            m_MemoryManager->GetIntraLayerManager());
        }
        case LayerType::LogicalBinary :
        {
            auto logicalBinaryQueueDescriptor = PolymorphicDowncast<const LogicalBinaryQueueDescriptor*>(&descriptor);
            switch(logicalBinaryQueueDescriptor->m_Parameters.m_Operation)
            {
                case LogicalBinaryOperation::LogicalAnd:
                    return std::make_unique<NeonLogicalAndWorkload>(*logicalBinaryQueueDescriptor, info);
                case LogicalBinaryOperation::LogicalOr:
                    return std::make_unique<NeonLogicalOrWorkload>(*logicalBinaryQueueDescriptor, info);
                default:
                    return nullptr;
            }
        }
        case LayerType::Lstm :
        {
            auto lstmQueueDescriptor = PolymorphicDowncast<const LstmQueueDescriptor*>(&descriptor);
            return MakeWorkloadHelper<NeonLstmFloatWorkload, NullWorkload>(*lstmQueueDescriptor, info);
        }
        case LayerType::Maximum :
        {
            auto maximumQueueDescriptor = PolymorphicDowncast<const MaximumQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonMaximumWorkload>(*maximumQueueDescriptor, info);
        }
        case LayerType::Mean :
        {
            auto meanQueueDescriptor = PolymorphicDowncast<const MeanQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonMeanWorkload>(*meanQueueDescriptor, info);
        }
        case LayerType::MemCopy :
        {
            auto memCopyQueueDescriptor = PolymorphicDowncast<const MemCopyQueueDescriptor*>(&descriptor);
            if (memCopyQueueDescriptor->m_Inputs.empty() || !memCopyQueueDescriptor->m_Inputs[0])
            {
                throw InvalidArgumentException("NeonWorkloadFactory: Invalid null input for MemCopy workload");
            }
            return MakeWorkloadHelper<CopyMemGenericWorkload, CopyMemGenericWorkload>(*memCopyQueueDescriptor, info);
        }
        case LayerType::MemImport :
        {
            auto memImportQueueDescriptor = PolymorphicDowncast<const MemImportQueueDescriptor*>(&descriptor);
            if (memImportQueueDescriptor->m_Inputs.empty() || !memImportQueueDescriptor->m_Inputs[0])
            {
                throw InvalidArgumentException("NeonWorkloadFactory: Invalid null input for MemImport workload");
            }
            return std::make_unique<ImportMemGenericWorkload>(*memImportQueueDescriptor, info);
        }
        case LayerType::Minimum :
        {
            auto minimumQueueDescriptor = PolymorphicDowncast<const MinimumQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonMinimumWorkload>(*minimumQueueDescriptor, info);
        }
        case LayerType::Multiplication :
        {
            auto multiplicationQueueDescriptor = PolymorphicDowncast<const MultiplicationQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonMultiplicationWorkload>(*multiplicationQueueDescriptor, info);
        }
        case LayerType::Normalization :
        {
            auto normalizationQueueDescriptor = PolymorphicDowncast<const NormalizationQueueDescriptor*>(&descriptor);
            return MakeWorkloadHelper<NeonNormalizationFloatWorkload, NullWorkload>
                (*normalizationQueueDescriptor, info, m_MemoryManager->GetIntraLayerManager());
        }
        case LayerType::Output :
        {
            auto outputQueueDescriptor = PolymorphicDowncast<const OutputQueueDescriptor*>(&descriptor);
            return std::make_unique<CopyMemGenericWorkload>(*outputQueueDescriptor, info);
        }
        case LayerType::Pad :
        {
            auto padQueueDescriptor = PolymorphicDowncast<const PadQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonPadWorkload>(*padQueueDescriptor, info);
        }
        case LayerType::Permute :
        {
            auto permuteQueueDescriptor = PolymorphicDowncast<const PermuteQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonPermuteWorkload>(*permuteQueueDescriptor, info);
        }
        case LayerType::Pooling2d :
        {
            auto pooling2dQueueDescriptor = PolymorphicDowncast<const Pooling2dQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonPooling2dWorkload>(*pooling2dQueueDescriptor, info);
        }
        case LayerType::Pooling3d :
        {
            auto pooling3dQueueDescriptor = PolymorphicDowncast<const Pooling3dQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonPooling3dWorkload>(*pooling3dQueueDescriptor, info);
        }
        case LayerType::PreCompiled :
        {
            auto preCompiledQueueDescriptor = PolymorphicDowncast<const PreCompiledQueueDescriptor*>(&descriptor);
            return MakeWorkloadHelper<NullWorkload, NullWorkload>(*preCompiledQueueDescriptor, info);
        }
        case LayerType::Prelu :
        {
            auto preluQueueDescriptor = PolymorphicDowncast<const PreluQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonPreluWorkload>(*preluQueueDescriptor, info);
        }
        case LayerType::QLstm :
        {
            auto qLstmQueueDescriptor = PolymorphicDowncast<const QLstmQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonQLstmWorkload>(*qLstmQueueDescriptor, info);
        }
        case LayerType::Quantize :
        {
            auto quantizeQueueDescriptor = PolymorphicDowncast<const QuantizeQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonQuantizeWorkload>(*quantizeQueueDescriptor, info);
        }
        case LayerType::QuantizedLstm :
        {
            auto quantizedLstmQueueDescriptor = PolymorphicDowncast<const QuantizedLstmQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonQuantizedLstmWorkload>(*quantizedLstmQueueDescriptor, info);
        }
        case LayerType::Rank :
        {
            auto rankQueueDescriptor = PolymorphicDowncast<const RankQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonRankWorkload>(*rankQueueDescriptor, info);
        }
        case LayerType::Reduce :
        {
            auto reduceQueueDescriptor = PolymorphicDowncast<const ReduceQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonReduceWorkload>(*reduceQueueDescriptor, info);
        }
        case LayerType::Reshape :
        {
            auto reshapeQueueDescriptor = PolymorphicDowncast<const ReshapeQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonReshapeWorkload>(*reshapeQueueDescriptor, info);
        }
        case LayerType::Resize :
        {
            auto resizeQueueDescriptor = PolymorphicDowncast<const ResizeQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonResizeWorkload>(*resizeQueueDescriptor, info);
        }
        case LayerType::Slice :
        {
            auto sliceQueueDescriptor = PolymorphicDowncast<const SliceQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonSliceWorkload>(*sliceQueueDescriptor, info);
        }
        case LayerType::Softmax :
        {
            auto softmaxQueueDescriptor = PolymorphicDowncast<const SoftmaxQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonSoftmaxWorkload>(*softmaxQueueDescriptor,
                                                         info,
                                                         m_MemoryManager->GetIntraLayerManager());
        }
        case LayerType::SpaceToBatchNd :
        {
            auto spaceToBatchNdQueueDescriptor
                     = PolymorphicDowncast<const SpaceToBatchNdQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonSpaceToBatchNdWorkload>(*spaceToBatchNdQueueDescriptor, info);
        }
        case LayerType::SpaceToDepth :
        {
            auto spaceToDepthQueueDescriptor = PolymorphicDowncast<const SpaceToDepthQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonSpaceToDepthWorkload>(*spaceToDepthQueueDescriptor, info);
        }
        case LayerType::Splitter :
        {
            auto splitterQueueDescriptor = PolymorphicDowncast<const SplitterQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonSplitterWorkload>(*splitterQueueDescriptor, info);
        }
        case LayerType::Stack :
        {
            auto stackQueueDescriptor = PolymorphicDowncast<const StackQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonStackWorkload>(*stackQueueDescriptor, info);
        }
        case LayerType::StridedSlice :
        {
            auto stridedSliceQueueDescriptor = PolymorphicDowncast<const StridedSliceQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonStridedSliceWorkload>(*stridedSliceQueueDescriptor, info);
        }
        case LayerType::Subtraction :
        {
            auto subtractionQueueDescriptor = PolymorphicDowncast<const SubtractionQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonSubtractionWorkload>(*subtractionQueueDescriptor, info);
        }
        case LayerType::Tile:
        {
            auto tileQueueDescriptor = PolymorphicDowncast<const TileQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonTileWorkload>(*tileQueueDescriptor, info);
        }
        case LayerType::Transpose :
        {
            auto transposeQueueDescriptor = PolymorphicDowncast<const TransposeQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonTransposeWorkload>(*transposeQueueDescriptor, info);
        }
        case LayerType::TransposeConvolution2d :
        {
            auto transposeConvolution2dQueueDescriptor
                     = PolymorphicDowncast<const TransposeConvolution2dQueueDescriptor*>(&descriptor);
            return std::make_unique<NeonTransposeConvolution2dWorkload>(*transposeConvolution2dQueueDescriptor,
                                                                        info,
                                                                        m_MemoryManager->GetIntraLayerManager());
        }
        case LayerType::UnidirectionalSequenceLstm :
        {
            auto desc = PolymorphicDowncast<const UnidirectionalSequenceLstmQueueDescriptor*>(&descriptor);
            if ((info.m_InputTensorInfos[0].GetDataType() == armnn::DataType::Float32) &&
                (info.m_InputTensorInfos[1].GetDataType() == armnn::DataType::Float32) &&
                (info.m_InputTensorInfos[2].GetDataType() == armnn::DataType::Float32) &&
                (info.m_OutputTensorInfos[0].GetDataType() == armnn::DataType::Float32) &&
                (info.m_OutputTensorInfos[1].GetDataType() == armnn::DataType::Float32) &&
                (info.m_OutputTensorInfos[2].GetDataType() == armnn::DataType::Float32))
            {
                return std::make_unique<NeonUnidirectionalSequenceLstmFloatWorkload>(*desc, info);
            }
            else
            {
                return std::make_unique<NeonUnidirectionalSequenceLstmWorkload>(*desc, info);
            }
        }
        default:
            return nullptr;
    }
}

} // namespace armnn
