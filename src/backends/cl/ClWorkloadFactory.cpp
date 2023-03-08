//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
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
#include <armnn/backends/MemCopyWorkload.hpp>
#include <backendsCommon/MemImportWorkload.hpp>
#include <armnn/backends/TensorHandle.hpp>

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

std::unique_ptr<IWorkload> ClWorkloadFactory::CreateWorkload(LayerType type,
                                                             const QueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    switch(type)
    {
        case LayerType::Activation :
        {
            auto activationQueueDescriptor = PolymorphicDowncast<const ActivationQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClActivationWorkload>(*activationQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Addition :
        {
            auto additionQueueDescriptor = PolymorphicDowncast<const AdditionQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClAdditionWorkload>(*additionQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::ArgMinMax :
        {
            auto argMinMaxQueueDescriptor = PolymorphicDowncast<const ArgMinMaxQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClArgMinMaxWorkload>(*argMinMaxQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::BatchMatMul :
        {
            auto batchMatMulQueueDescriptor = PolymorphicDowncast<const BatchMatMulQueueDescriptor*>(&descriptor);
            return std::make_unique<ClBatchMatMulWorkload>(*batchMatMulQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::BatchNormalization :
        {
            auto batchNormalizationQueueDescriptor
                    = PolymorphicDowncast<const BatchNormalizationQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClBatchNormalizationFloatWorkload, NullWorkload>
                    (*batchNormalizationQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::BatchToSpaceNd :
        {
            auto batchToSpaceNdQueueDescriptor
                    = PolymorphicDowncast<const BatchToSpaceNdQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClBatchToSpaceNdWorkload>(*batchToSpaceNdQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Cast :
        {
            auto castQueueDescriptor = PolymorphicDowncast<const CastQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClCastWorkload>(*castQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::ChannelShuffle :
        {
            auto channelShuffleQueueDescriptor
                    = PolymorphicDowncast<const ChannelShuffleQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClChannelShuffleWorkload>(*channelShuffleQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Comparison :
        {
            auto comparisonQueueDescriptor = PolymorphicDowncast<const ComparisonQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClComparisonWorkload>(*comparisonQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Concat :
        {
            auto concatQueueDescriptor = PolymorphicDowncast<const ConcatQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClConcatWorkload>(*concatQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Constant :
        {
            auto constantQueueDescriptor = PolymorphicDowncast<const ConstantQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClConstantWorkload>(*constantQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::ConvertFp16ToFp32 :
        {
            auto convertFp16ToFp32QueueDescriptor
                    = PolymorphicDowncast<const ConvertFp16ToFp32QueueDescriptor*>(&descriptor);
            return MakeWorkload<ClConvertFp16ToFp32Workload>(*convertFp16ToFp32QueueDescriptor,
                                                             info,
                                                             m_CLCompileContext);
        }
        case LayerType::ConvertFp32ToFp16 :
        {
            auto convertFp32ToFp16QueueDescriptor
                    = PolymorphicDowncast<const ConvertFp32ToFp16QueueDescriptor*>(&descriptor);
            return MakeWorkload<ClConvertFp32ToFp16Workload>(*convertFp32ToFp16QueueDescriptor,
                                                             info,
                                                             m_CLCompileContext);
        }
        case LayerType::Convolution2d :
        {
            auto convolution2dQueueDescriptor = PolymorphicDowncast<const Convolution2dQueueDescriptor*>(&descriptor);

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
            return MakeWorkload<ClConvolution2dWorkload>(*convolution2dQueueDescriptor,
                                                         info,
                                                         m_MemoryManager->GetIntraLayerManager(),
                                                         m_CLCompileContext,
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
                    auto modelOptions = dynamic_cast<ClBackendModelContext*>(m_ModelContextPtr.get());
                    if (modelOptions)
                    {
                        isFastMathEnabled = modelOptions->IsFastMathEnabled();
                    }
                }
            }
            return MakeWorkload<ClConvolution3dWorkload>(*convolution3dQueueDescriptor,
                                                         info,
                                                         m_MemoryManager->GetIntraLayerManager(),
                                                         m_CLCompileContext,
                                                         isFastMathEnabled);
        }
        case LayerType::Debug :
        {
            auto debugQueueDescriptor = PolymorphicDowncast<const DebugQueueDescriptor*>(&descriptor);
            return MakeWorkload<NullWorkload, NullWorkload>(*debugQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::DepthToSpace :
        {
            auto depthToSpaceQueueDescriptor = PolymorphicDowncast<const DepthToSpaceQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClDepthToSpaceWorkload>(*depthToSpaceQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::DepthwiseConvolution2d :
        {
            auto depthwiseConvolution2dQueueDescriptor
                    = PolymorphicDowncast<const DepthwiseConvolution2dQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClDepthwiseConvolutionWorkload>(*depthwiseConvolution2dQueueDescriptor,
                                                                info,
                                                                m_CLCompileContext);
        }
        case LayerType::Dequantize :
        {
            auto dequantizeQueueDescriptor = PolymorphicDowncast<const DequantizeQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClDequantizeWorkload>(*dequantizeQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::DetectionPostProcess :
        {
            auto detectionPostProcessQueueDescriptor
                    = PolymorphicDowncast<const DetectionPostProcessQueueDescriptor*>(&descriptor);
            return MakeWorkload<NullWorkload, NullWorkload>(*detectionPostProcessQueueDescriptor,
                                                            info,
                                                            m_CLCompileContext);
        }
        case LayerType::Division :
        {
            auto divisionQueueDescriptor = PolymorphicDowncast<const DivisionQueueDescriptor*>(&descriptor);
            return std::make_unique<ClDivisionWorkload>(*divisionQueueDescriptor, info, m_CLCompileContext);
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
                    additionQueueDescriptor.m_AdditionalInfoObject =
                            elementwiseBinaryQueueDescriptor->m_AdditionalInfoObject;
                    return std::make_unique<ClAdditionWorkload>(additionQueueDescriptor, info, m_CLCompileContext);
                }
                case BinaryOperation::Div:
                {
                    DivisionQueueDescriptor divisionQueueDescriptor;
                    divisionQueueDescriptor.m_Inputs = descriptor.m_Inputs;
                    divisionQueueDescriptor.m_Outputs = descriptor.m_Outputs;
                    divisionQueueDescriptor.m_AdditionalInfoObject =
                            elementwiseBinaryQueueDescriptor->m_AdditionalInfoObject;
                    return std::make_unique<ClDivisionWorkload>(divisionQueueDescriptor, info, m_CLCompileContext);
                }
                case BinaryOperation::Maximum:
                {
                    MaximumQueueDescriptor maximumQueueDescriptor;
                    maximumQueueDescriptor.m_Inputs = descriptor.m_Inputs;
                    maximumQueueDescriptor.m_Outputs = descriptor.m_Outputs;
                    maximumQueueDescriptor.m_AdditionalInfoObject =
                            elementwiseBinaryQueueDescriptor->m_AdditionalInfoObject;
                    return std::make_unique<ClMaximumWorkload>(maximumQueueDescriptor, info, m_CLCompileContext);
                }
                case BinaryOperation::Minimum:
                {
                    MinimumQueueDescriptor minimumQueueDescriptor;
                    minimumQueueDescriptor.m_Inputs = descriptor.m_Inputs;
                    minimumQueueDescriptor.m_Outputs = descriptor.m_Outputs;
                    minimumQueueDescriptor.m_AdditionalInfoObject =
                            elementwiseBinaryQueueDescriptor->m_AdditionalInfoObject;
                    return std::make_unique<ClMinimumWorkload>(minimumQueueDescriptor, info, m_CLCompileContext);
                }
                case BinaryOperation::Mul:
                {
                    MultiplicationQueueDescriptor multiplicationQueueDescriptor;
                    multiplicationQueueDescriptor.m_Inputs = descriptor.m_Inputs;
                    multiplicationQueueDescriptor.m_Outputs = descriptor.m_Outputs;
                    multiplicationQueueDescriptor.m_AdditionalInfoObject =
                            elementwiseBinaryQueueDescriptor->m_AdditionalInfoObject;
                    return std::make_unique<ClMultiplicationWorkload>(multiplicationQueueDescriptor,
                                                                      info,
                                                                      m_CLCompileContext);
                }
                case BinaryOperation::Sub:
                {
                    SubtractionQueueDescriptor subtractionQueueDescriptor;
                    subtractionQueueDescriptor.m_Inputs = descriptor.m_Inputs;
                    subtractionQueueDescriptor.m_Outputs = descriptor.m_Outputs;
                    subtractionQueueDescriptor.m_AdditionalInfoObject =
                            elementwiseBinaryQueueDescriptor->m_AdditionalInfoObject;
                    return std::make_unique<ClSubtractionWorkload>(subtractionQueueDescriptor,
                                                                   info,
                                                                   m_CLCompileContext);
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

                    return  std::make_unique<ClAbsWorkload>(absQueueDescriptor, info, m_CLCompileContext);
                }
                case UnaryOperation::Exp:
                    return std::make_unique<ClExpWorkload>(*elementwiseUnaryQueueDescriptor, info, m_CLCompileContext);
                case UnaryOperation::Log:
                    return std::make_unique<ClLogWorkload>(*elementwiseUnaryQueueDescriptor, info, m_CLCompileContext);
                case UnaryOperation::LogicalNot:
                    return std::make_unique<ClLogicalNotWorkload>(*elementwiseUnaryQueueDescriptor,
                                                                  info,
                                                                  m_CLCompileContext);
                case UnaryOperation::Neg:
                    return std::make_unique<ClNegWorkload>(*elementwiseUnaryQueueDescriptor, info, m_CLCompileContext);
                case UnaryOperation::Rsqrt:
                {
                    RsqrtQueueDescriptor rsqrtQueueDescriptor;
                    rsqrtQueueDescriptor.m_Inputs  = elementwiseUnaryQueueDescriptor->m_Inputs;
                    rsqrtQueueDescriptor.m_Outputs = elementwiseUnaryQueueDescriptor->m_Outputs;

                    return std::make_unique<ClRsqrtWorkload>(rsqrtQueueDescriptor, info, m_CLCompileContext);
                }
                case UnaryOperation::Sin:
                    return std::make_unique<ClSinWorkload>(*elementwiseUnaryQueueDescriptor, info, m_CLCompileContext);
                case UnaryOperation::Sqrt:
                    return std::make_unique<ClSqrtWorkload>(*elementwiseUnaryQueueDescriptor, info, m_CLCompileContext);
                default:
                    return nullptr;
            }
        }
        case LayerType::Fill :
        {
            auto fillQueueDescriptor = PolymorphicDowncast<const FillQueueDescriptor*>(&descriptor);
            return std::make_unique<ClFillWorkload>(*fillQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Floor :
        {
            auto floorQueueDescriptor = PolymorphicDowncast<const FloorQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClFloorFloatWorkload, NullWorkload>(*floorQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::FullyConnected :
        {
            auto fullyConnectedQueueDescriptor
                    = PolymorphicDowncast<const FullyConnectedQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClFullyConnectedWorkload>(*fullyConnectedQueueDescriptor,
                                                          info,
                                                          m_MemoryManager->GetIntraLayerManager(),
                                                          m_CLCompileContext);
        }
        case LayerType::Gather :
        {
            auto gatherQueueDescriptor = PolymorphicDowncast<const GatherQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClGatherWorkload>(*gatherQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::GatherNd :
        {
            auto gatherNdQueueDescriptor = PolymorphicDowncast<const GatherNdQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClGatherNdWorkload>(*gatherNdQueueDescriptor, info, m_CLCompileContext);
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
            return MakeWorkload<ClInstanceNormalizationWorkload>(*instanceNormalizationQueueDescriptor,
                                                                 info,
                                                                 m_CLCompileContext);
        }
        case LayerType::L2Normalization :
        {
            auto l2NormalizationQueueDescriptor
                    = PolymorphicDowncast<const L2NormalizationQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClL2NormalizationFloatWorkload, NullWorkload>(*l2NormalizationQueueDescriptor,
                                                                              info,
                                                                              m_CLCompileContext);
        }
        case LayerType::LogicalBinary :
        {
            auto logicalBinaryQueueDescriptor = PolymorphicDowncast<const LogicalBinaryQueueDescriptor*>(&descriptor);

            switch(logicalBinaryQueueDescriptor->m_Parameters.m_Operation)
            {
                case LogicalBinaryOperation::LogicalAnd:
                    return std::make_unique<ClLogicalAndWorkload>(*logicalBinaryQueueDescriptor,
                                                                  info,
                                                                  m_CLCompileContext);
                case LogicalBinaryOperation::LogicalOr:
                    return std::make_unique<ClLogicalOrWorkload>(*logicalBinaryQueueDescriptor,
                                                                 info,
                                                                 m_CLCompileContext);
                default:
                    return nullptr;
            }
        }
        case LayerType::LogSoftmax :
        {
            auto logSoftmaxQueueDescriptor = PolymorphicDowncast<const LogSoftmaxQueueDescriptor*>(&descriptor);

            return MakeWorkload<ClLogSoftmaxWorkload>(*logSoftmaxQueueDescriptor,
                                                      info,
                                                      m_MemoryManager->GetIntraLayerManager(),
                                                      m_CLCompileContext);
        }
        case LayerType::Lstm :
        {
            auto lstmQueueDescriptor = PolymorphicDowncast<const LstmQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClLstmFloatWorkload, NullWorkload>(*lstmQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Maximum :
        {
            auto maximumQueueDescriptor = PolymorphicDowncast<const MaximumQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClMaximumWorkload>(*maximumQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Mean :
        {
            auto meanQueueDescriptor = PolymorphicDowncast<const MeanQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClMeanWorkload>(*meanQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::MemCopy :
        {
            auto memCopyQueueDescriptor = PolymorphicDowncast<const MemCopyQueueDescriptor*>(&descriptor);
            if (memCopyQueueDescriptor->m_Inputs.empty() || !memCopyQueueDescriptor->m_Inputs[0])
            {
                throw InvalidArgumentException("ClWorkloadFactory: Invalid null input for MemCopy workload");
            }
            return MakeWorkload<CopyMemGenericWorkload>(*memCopyQueueDescriptor, info);
        }
        case LayerType::MemImport :
        {
            auto memImportQueueDescriptor = PolymorphicDowncast<const MemImportQueueDescriptor*>(&descriptor);
            if (memImportQueueDescriptor->m_Inputs.empty() || !memImportQueueDescriptor->m_Inputs[0])
            {
                throw InvalidArgumentException("ClWorkloadFactory: Invalid null input for MemImport workload");
            }
            return std::make_unique<ImportMemGenericWorkload>(*memImportQueueDescriptor, info);
        }
        case LayerType::Minimum :
        {
            auto minimumQueueDescriptor = PolymorphicDowncast<const MinimumQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClMinimumWorkload>(*minimumQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Multiplication :
        {
            auto multiplicationQueueDescriptor = PolymorphicDowncast<const MultiplicationQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClMultiplicationWorkload>(*multiplicationQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Normalization :
        {
            auto normalizationQueueDescriptor = PolymorphicDowncast<const NormalizationQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClNormalizationFloatWorkload, NullWorkload>(*normalizationQueueDescriptor,
                                                                            info,
                                                                            m_CLCompileContext);
        }
        case LayerType::Output :
        {
            auto outputQueueDescriptor = PolymorphicDowncast<const OutputQueueDescriptor*>(&descriptor);
            return std::make_unique<CopyMemGenericWorkload>(*outputQueueDescriptor, info);
        }
        case LayerType::Pad :
        {
            auto padQueueDescriptor = PolymorphicDowncast<const PadQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClPadWorkload>(*padQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Permute :
        {
            auto permuteQueueDescriptor = PolymorphicDowncast<const PermuteQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClPermuteWorkload>(*permuteQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Pooling2d :
        {
            auto pooling2dQueueDescriptor = PolymorphicDowncast<const Pooling2dQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClPooling2dWorkload>(*pooling2dQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Pooling3d :
        {
            auto pooling3dQueueDescriptor = PolymorphicDowncast<const Pooling3dQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClPooling3dWorkload>(*pooling3dQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::PreCompiled :
        {
            auto preCompiledQueueDescriptor = PolymorphicDowncast<const PreCompiledQueueDescriptor*>(&descriptor);
            return MakeWorkload<NullWorkload, NullWorkload>(*preCompiledQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Prelu :
        {
            auto preluQueueDescriptor = PolymorphicDowncast<const PreluQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClPreluWorkload>(*preluQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::QLstm :
        {
            auto qLstmQueueDescriptor = PolymorphicDowncast<const QLstmQueueDescriptor*>(&descriptor);
            return std::make_unique<ClQLstmWorkload>(*qLstmQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Quantize :
        {
            auto quantizeQueueDescriptor = PolymorphicDowncast<const QuantizeQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClQuantizeWorkload>(*quantizeQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::QuantizedLstm :
        {
            auto quantizedLstmQueueDescriptor = PolymorphicDowncast<const QuantizedLstmQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClQuantizedLstmWorkload>(*quantizedLstmQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Rank :
        {
            auto rankQueueDescriptor = PolymorphicDowncast<const RankQueueDescriptor*>(&descriptor);
            return std::make_unique<ClRankWorkload>(*rankQueueDescriptor, info);
        }
        case LayerType::Reduce :
        {
            auto reduceQueueDescriptor = PolymorphicDowncast<const ReduceQueueDescriptor*>(&descriptor);
            return std::make_unique<ClReduceWorkload>(*reduceQueueDescriptor, info);
        }
        case LayerType::Reshape :
        {
            auto reshapeQueueDescriptor = PolymorphicDowncast<const ReshapeQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClReshapeWorkload>(*reshapeQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Resize :
        {
            auto resizeQueueDescriptor = PolymorphicDowncast<const ResizeQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClResizeWorkload>(*resizeQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Slice :
        {
            auto sliceQueueDescriptor = PolymorphicDowncast<const SliceQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClSliceWorkload>(*sliceQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Softmax :
        {
            auto softmaxQueueDescriptor = PolymorphicDowncast<const SoftmaxQueueDescriptor*>(&descriptor);
            return std::make_unique<ClSoftmaxWorkload>(*softmaxQueueDescriptor,
                                                       info,
                                                       m_MemoryManager->GetIntraLayerManager(),
                                                       m_CLCompileContext);
        }
        case LayerType::SpaceToBatchNd :
        {
            auto spaceToBatchNdQueueDescriptor
                    = PolymorphicDowncast<const SpaceToBatchNdQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClSpaceToBatchNdWorkload>(*spaceToBatchNdQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::SpaceToDepth :
        {
            auto spaceToDepthQueueDescriptor = PolymorphicDowncast<const SpaceToDepthQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClSpaceToDepthWorkload>(*spaceToDepthQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Splitter :
        {
            auto splitterQueueDescriptor = PolymorphicDowncast<const SplitterQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClSplitterWorkload>(*splitterQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Stack :
        {
            auto stackQueueDescriptor = PolymorphicDowncast<const StackQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClStackWorkload>(*stackQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::StridedSlice :
        {
            auto stridedSliceQueueDescriptor = PolymorphicDowncast<const StridedSliceQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClStridedSliceWorkload>(*stridedSliceQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Subtraction :
        {
            auto subtractionQueueDescriptor = PolymorphicDowncast<const SubtractionQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClSubtractionWorkload>(*subtractionQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Transpose :
        {
            auto transposeQueueDescriptor = PolymorphicDowncast<const TransposeQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClTransposeWorkload>(*transposeQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::TransposeConvolution2d :
        {
            auto transposeConvolution2dQueueDescriptor
                    = PolymorphicDowncast<const TransposeConvolution2dQueueDescriptor*>(&descriptor);
            return MakeWorkload<ClTransposeConvolution2dWorkload>(*transposeConvolution2dQueueDescriptor,
                                                                  info,
                                                                  m_MemoryManager->GetIntraLayerManager(),
                                                                  m_CLCompileContext);
        }
        case LayerType::UnidirectionalSequenceLstm :
        {
            auto desc = PolymorphicDowncast<const UnidirectionalSequenceLstmQueueDescriptor*>(&descriptor);
            return MakeWorkloadHelper<ClUnidirectionalSequenceLstmFloatWorkload, NullWorkload>(*desc,
                                                                                               info,
                                                                                               m_CLCompileContext);
        }
        default:
            return nullptr;
    }
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
