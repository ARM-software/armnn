//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ClWorkloadFactory.hpp"
#include "ClBackendId.hpp"
#include "ClBackendModelContext.hpp"
#include "ClContextDeserializer.hpp"
#include "ClContextSerializer.hpp"

#include <Layer.hpp>

#include <armnn/Exceptions.hpp>
#include <armnn/Logging.hpp>
#include <armnn/Utils.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <backendsCommon/MakeWorkloadHelper.hpp>
#include <backendsCommon/MemCopyWorkload.hpp>
#include <backendsCommon/MemImportWorkload.hpp>
#include <backendsCommon/TensorHandle.hpp>

#include <cl/ClTensorHandle.hpp>
#include <cl/workloads/ClWorkloads.hpp>
#include <cl/workloads/ClWorkloadUtils.hpp>

#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/runtime/CL/CLBufferAllocator.h>
#include <arm_compute/runtime/CL/CLScheduler.h>

#include <armnnUtils/Filesystem.hpp>
#include <fstream>

#include <sys/stat.h>

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

bool ClWorkloadFactory::IsLayerSupported(const IConnectableLayer& layer,
                                         Optional<DataType> dataType,
                                         std::string& outReasonIfUnsupported,
                                         const ModelOptions& modelOptions)
{
    return IWorkloadFactory::IsLayerSupported(s_Id, layer, dataType, outReasonIfUnsupported, modelOptions);
}

const BackendId& ClWorkloadFactory::GetBackendId() const
{
    return s_Id;
}

void ClWorkloadFactory::AfterWorkloadsCreated()
{
    if(m_ModelContextPtr)
    {
        auto modelOptions = dynamic_cast<ClBackendModelContext*>(m_ModelContextPtr.get());
        if (modelOptions->SaveCachedNetwork())
        {
            ClContextSerializer serializer;
            serializer.Serialize(m_CLCompileContext);
            auto cachedFd = modelOptions->GetCachedFileDescriptor();
            if (cachedFd != -1)
            {
                std::vector<uint8_t> compiledContextData;
                std::stringstream stream;
                bool serialized = serializer.SaveSerializedToStream(stream);
                if (serialized)
                {
                    std::string const serializedString{stream.str()};
                    std::copy(serializedString.begin(),
                              serializedString.end(),
                              std::back_inserter(compiledContextData));
                    auto success = write(cachedFd, compiledContextData.data(), compiledContextData.size());
                    if (success == -1)
                    {
                        ARMNN_LOG(info) << "ClWorkloadFactory:: Could not cache the compiled context!";
                    }
                }
            }

            // Save map to a filepath provided in ModelOptions
            auto filePath = modelOptions->GetCachedNetworkFilePath();
            if (filePath != "" && fs::exists(filePath) && fs::is_regular_file(filePath))
            {
                // Serialize ClContext to the file specified
                std::ofstream file(filePath, std::ios::out | std::ios::binary);
                serializer.SaveSerializedToStream(file);
            }
        }
    }
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

void ClWorkloadFactory::InitializeCLCompileContext()
{
    // Initialize our m_CLCompileContext using default device and context
    auto context = arm_compute::CLKernelLibrary::get().context();
    auto device  = arm_compute::CLKernelLibrary::get().get_device();
    m_CLCompileContext = arm_compute::CLCompileContext(context, device);

    if (m_ModelContextPtr)
    {
        // Load saved programs if the user has set a filepath
        auto modelOptions = dynamic_cast<ClBackendModelContext*>(m_ModelContextPtr.get());
        auto filePath = modelOptions->GetCachedNetworkFilePath();
        if (!(modelOptions->SaveCachedNetwork()))
        {
            ClContextDeserializer deserializer;
            auto cachedFd = modelOptions->GetCachedFileDescriptor();
            if (cachedFd != -1)
            {
                struct stat statBuffer;
                if (fstat(cachedFd, &statBuffer) == 0)
                {
                    long dataSize = static_cast<long>(statBuffer.st_size);
                    if( dataSize > 0)
                    {
                        auto offset = lseek(cachedFd, 0, SEEK_CUR);
                        if (offset == 0)
                        {
                            std::vector <uint8_t> compiledContextData(static_cast<unsigned int>(dataSize));
                            auto success = pread(cachedFd, compiledContextData.data(), compiledContextData.size(), 0);
                            if (success != -1)
                            {
                                deserializer.DeserializeFromBinary(m_CLCompileContext,
                                                                   context,
                                                                   device,
                                                                   compiledContextData);
                            }
                        }
                    }

                }
            }

            if (filePath != "" && fs::exists(filePath) && fs::is_regular_file(filePath))
            {
                // Deserialize binary file and load into m_CLCompileContext
                deserializer.Deserialize(m_CLCompileContext, context, device, filePath);
            }
        }
    }
}

ClWorkloadFactory::ClWorkloadFactory(const std::shared_ptr<ClMemoryManager>& memoryManager)
    : m_MemoryManager(memoryManager), m_ModelContextPtr(IBackendInternal::IBackendSpecificModelContextPtr{})
{
    InitializeCLCompileContext();
}

ClWorkloadFactory::ClWorkloadFactory(const std::shared_ptr<ClMemoryManager>& memoryManager,
                                     const IBackendInternal::IBackendSpecificModelContextPtr& modelContextPtr)
    : m_MemoryManager(memoryManager), m_ModelContextPtr(modelContextPtr)
{
    InitializeCLCompileContext();
}

std::unique_ptr<ITensorHandle> ClWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                     const bool IsMemoryManaged) const
{
    IgnoreUnused(IsMemoryManaged);
    std::unique_ptr<ClTensorHandle> tensorHandle = std::make_unique<ClTensorHandle>(tensorInfo);
    tensorHandle->SetMemoryGroup(m_MemoryManager->GetInterLayerMemoryGroup());

    return tensorHandle;
}

std::unique_ptr<ITensorHandle> ClWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                     DataLayout dataLayout,
                                                                     const bool IsMemoryManaged) const
{
    IgnoreUnused(IsMemoryManaged);
    std::unique_ptr<ClTensorHandle> tensorHandle = std::make_unique<ClTensorHandle>(tensorInfo, dataLayout);
    tensorHandle->SetMemoryGroup(m_MemoryManager->GetInterLayerMemoryGroup());

    return tensorHandle;
}

std::unique_ptr<ITensorHandle> ClWorkloadFactory::CreateSubTensorHandle(ITensorHandle& parent,
                                                                        TensorShape const& subTensorShape,
                                                                        unsigned int const* subTensorOrigin) const
{
    arm_compute::Coordinates coords;
    arm_compute::TensorShape shape = armcomputetensorutils::BuildArmComputeTensorShape(subTensorShape);

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

    return std::make_unique<ClSubTensorHandle>(
        PolymorphicDowncast<IClTensorHandle*>(&parent), shape, coords);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateActivation(const ActivationQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    return MakeWorkload<ClActivationWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    return MakeWorkload<ClAdditionWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateArgMinMax(const ArgMinMaxQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    return std::make_unique<ClArgMinMaxWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateBatchNormalization(
    const BatchNormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClBatchNormalizationFloatWorkload, NullWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateBatchToSpaceNd(const BatchToSpaceNdQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const
{
    return MakeWorkload<ClBatchToSpaceNdWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateCast(const CastQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info) const
{
    return MakeWorkload<ClCastWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateChannelShuffle(const ChannelShuffleQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const
{
    return MakeWorkload<ClChannelShuffleWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateComparison(const ComparisonQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    return MakeWorkload<ClComparisonWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConcat(const ConcatQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return MakeWorkload<ClConcatWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConstant(const ConstantQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    return MakeWorkload<ClConstantWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConvertFp16ToFp32(
    const ConvertFp16ToFp32QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClConvertFp16ToFp32Workload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConvertFp32ToFp16(
    const ConvertFp32ToFp16QueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClConvertFp32ToFp16Workload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConvolution2d(const Convolution2dQueueDescriptor& descriptor,
                                                                  const WorkloadInfo& info) const
{
    bool isFastMathEnabled = false;
    if (m_ModelContextPtr)
    {
        if (m_ModelContextPtr.get() != nullptr)
        {
            auto modelOptions = dynamic_cast<ClBackendModelContext*>(m_ModelContextPtr.get());
            if (modelOptions)
            {
                isFastMathEnabled = modelOptions->IsFastMathEnabled();
            }
        }
    }
    return MakeWorkload<ClConvolution2dWorkload>(descriptor,
                                                 info,
                                                 m_MemoryManager->GetIntraLayerManager(),
                                                 m_CLCompileContext,
                                                 isFastMathEnabled);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateConvolution3d(const Convolution3dQueueDescriptor& descriptor,
                                                                  const WorkloadInfo& info) const
{
    bool isFastMathEnabled = false;
    if (m_ModelContextPtr)
    {
        if (m_ModelContextPtr.get() != nullptr)
        {
            auto modelOptions = dynamic_cast<ClBackendModelContext*>(m_ModelContextPtr.get());
            if (modelOptions)
            {
                isFastMathEnabled = modelOptions->IsFastMathEnabled();
            }
        }
    }
    return MakeWorkload<ClConvolution3dWorkload>(descriptor,
                                                 info,
                                                 m_MemoryManager->GetIntraLayerManager(),
                                                 m_CLCompileContext,
                                                 isFastMathEnabled);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateDebug(const DebugQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateDepthToSpace(const DepthToSpaceQueueDescriptor& descriptor,
                                                                 const WorkloadInfo& info) const
{
    return MakeWorkload<ClDepthToSpaceWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateDepthwiseConvolution2d(
    const DepthwiseConvolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClDepthwiseConvolutionWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateDequantize(const DequantizeQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    return MakeWorkload<ClDequantizeWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateDetectionPostProcess(
    const DetectionPostProcessQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateDivision(const DivisionQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    return std::make_unique<ClDivisionWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateElementwiseUnary(const ElementwiseUnaryQueueDescriptor& descriptor,
                                                                     const WorkloadInfo& info) const
{
    switch(descriptor.m_Parameters.m_Operation)
    {
        case UnaryOperation::Abs:
        {
            AbsQueueDescriptor absQueueDescriptor;
            absQueueDescriptor.m_Inputs  = descriptor.m_Inputs;
            absQueueDescriptor.m_Outputs = descriptor.m_Outputs;

            return  std::make_unique<ClAbsWorkload>(absQueueDescriptor, info, m_CLCompileContext);
        }
        case UnaryOperation::Exp:
            return std::make_unique<ClExpWorkload>(descriptor, info, m_CLCompileContext);
         case UnaryOperation::Log:
            return std::make_unique<ClLogWorkload>(descriptor, info, m_CLCompileContext);
        case UnaryOperation::LogicalNot:
            return std::make_unique<ClLogicalNotWorkload>(descriptor, info, m_CLCompileContext);
        case UnaryOperation::Neg:
            return std::make_unique<ClNegWorkload>(descriptor, info, m_CLCompileContext);
        case UnaryOperation::Rsqrt:
        {
            RsqrtQueueDescriptor rsqrtQueueDescriptor;
            rsqrtQueueDescriptor.m_Inputs  = descriptor.m_Inputs;
            rsqrtQueueDescriptor.m_Outputs = descriptor.m_Outputs;

            return std::make_unique<ClRsqrtWorkload>(rsqrtQueueDescriptor, info, m_CLCompileContext);
        }
        case UnaryOperation::Sin:
            return std::make_unique<ClSinWorkload>(descriptor, info, m_CLCompileContext);
        default:
            return nullptr;
    }
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateFill(const FillQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info) const
{
    return std::make_unique<ClFillWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateFloor(const FloorQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return MakeWorkload<ClFloorFloatWorkload, NullWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateFullyConnected(const FullyConnectedQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const
{
    return MakeWorkload<ClFullyConnectedWorkload>(descriptor,
                                                  info,
                                                  m_MemoryManager->GetIntraLayerManager(),
                                                  m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateGather(const GatherQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return MakeWorkload<ClGatherWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateInstanceNormalization(
    const InstanceNormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClInstanceNormalizationWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateL2Normalization(const L2NormalizationQueueDescriptor& descriptor,
                                                                    const WorkloadInfo& info) const
{
    return MakeWorkload<ClL2NormalizationFloatWorkload, NullWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateLogicalBinary(const LogicalBinaryQueueDescriptor& descriptor,
                                                                  const WorkloadInfo& info) const
{
    switch(descriptor.m_Parameters.m_Operation)
    {
        case LogicalBinaryOperation::LogicalAnd:
            return std::make_unique<ClLogicalAndWorkload>(descriptor, info, m_CLCompileContext);
        case LogicalBinaryOperation::LogicalOr:
            return std::make_unique<ClLogicalOrWorkload>(descriptor, info, m_CLCompileContext);
        default:
            return nullptr;
    }
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateLogSoftmax(const LogSoftmaxQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    return MakeWorkload<ClLogSoftmaxWorkload>(descriptor,
                                              info,
                                              m_MemoryManager->GetIntraLayerManager(),
                                              m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateLstm(const LstmQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info) const
{
    return MakeWorkload<ClLstmFloatWorkload, NullWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateMaximum(const MaximumQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return MakeWorkload<ClMaximumWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateMean(const MeanQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info) const
{
    return MakeWorkload<ClMeanWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    if (descriptor.m_Inputs.empty() || !descriptor.m_Inputs[0])
    {
        throw InvalidArgumentException("ClWorkloadFactory: Invalid null input for MemCopy workload");
    }

    return MakeWorkload<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateMemImport(const MemImportQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    if (descriptor.m_Inputs.empty() || !descriptor.m_Inputs[0])
    {
        throw InvalidArgumentException("ClWorkloadFactory: Invalid null input for MemImport workload");
    }

    return std::make_unique<ImportMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateMinimum(const MinimumQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return MakeWorkload<ClMinimumWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateMultiplication(const MultiplicationQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const
{
    return MakeWorkload<ClMultiplicationWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateNormalization(const NormalizationQueueDescriptor& descriptor,
                                                                  const WorkloadInfo& info) const
{
    return MakeWorkload<ClNormalizationFloatWorkload, NullWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateOutput(const OutputQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreatePad(const PadQueueDescriptor& descriptor,
                                                        const WorkloadInfo& info) const
{
    return MakeWorkload<ClPadWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreatePermute(const PermuteQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return MakeWorkload<ClPermuteWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreatePooling2d(const Pooling2dQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    return MakeWorkload<ClPooling2dWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreatePreCompiled(const PreCompiledQueueDescriptor& descriptor,
                                                                const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreatePrelu(const PreluQueueDescriptor &descriptor,
                                                          const WorkloadInfo &info) const
{
    return MakeWorkload<ClPreluWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateQLstm(const QLstmQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return std::make_unique<ClQLstmWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateQuantize(const QuantizeQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    return MakeWorkload<ClQuantizeWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateQuantizedLstm(const QuantizedLstmQueueDescriptor& descriptor,
                                                                  const WorkloadInfo& info) const
{
    return MakeWorkload<ClQuantizedLstmWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateRank(const RankQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info) const
{
    return std::make_unique<ClRankWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateReduce(const ReduceQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return std::make_unique<ClReduceWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateReshape(const ReshapeQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return MakeWorkload<ClReshapeWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateResize(const ResizeQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return MakeWorkload<ClResizeWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSlice(const SliceQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return MakeWorkload<ClSliceWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSoftmax(const SoftmaxQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return std::make_unique<ClSoftmaxWorkload>(descriptor,
                                               info,
                                               m_MemoryManager->GetIntraLayerManager(),
                                               m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSpaceToBatchNd(const SpaceToBatchNdQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const
{
    return MakeWorkload<ClSpaceToBatchNdWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSpaceToDepth(const SpaceToDepthQueueDescriptor& descriptor,
                                                                 const WorkloadInfo& info) const
{
    return MakeWorkload<ClSpaceToDepthWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSplitter(const SplitterQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    return MakeWorkload<ClSplitterWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateStack(const StackQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return MakeWorkload<ClStackWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateStridedSlice(const StridedSliceQueueDescriptor& descriptor,
                                                                 const WorkloadInfo& info) const
{
    return MakeWorkload<ClStridedSliceWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateSubtraction(const SubtractionQueueDescriptor& descriptor,
                                                                const WorkloadInfo& info) const
{
    return MakeWorkload<ClSubtractionWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateTranspose(const TransposeQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    return MakeWorkload<ClTransposeWorkload>(descriptor, info, m_CLCompileContext);
}

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateTransposeConvolution2d(
    const TransposeConvolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info) const
{
    return MakeWorkload<ClTransposeConvolution2dWorkload>(descriptor,
                                                          info,
                                                          m_MemoryManager->GetIntraLayerManager(),
                                                          m_CLCompileContext);
}

} // namespace armnn
