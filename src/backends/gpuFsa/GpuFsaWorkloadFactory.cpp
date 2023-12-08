//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <Layer.hpp>

#include "GpuFsaWorkloadFactory.hpp"
#include "GpuFsaBackendId.hpp"
#include "GpuFsaTensorHandle.hpp"

#include "workloads/GpuFsaConstantWorkload.hpp"
#include "workloads/GpuFsaPreCompiledWorkload.hpp"

#include <armnn/backends/MemCopyWorkload.hpp>

namespace armnn
{

namespace
{
static const BackendId s_Id{GpuFsaBackendId()};
}
template <typename QueueDescriptorType>
std::unique_ptr<IWorkload> GpuFsaWorkloadFactory::MakeWorkload(const QueueDescriptorType& /*descriptor*/,
                                                               const WorkloadInfo& /*info*/) const
{
    return nullptr;
}

template <DataType ArmnnType>
bool IsDataType(const WorkloadInfo& info)
{
    auto checkType = [](const TensorInfo& tensorInfo) {return tensorInfo.GetDataType() == ArmnnType;};
    auto it = std::find_if(std::begin(info.m_InputTensorInfos), std::end(info.m_InputTensorInfos), checkType);
    if (it != std::end(info.m_InputTensorInfos))
    {
        return true;
    }
    it = std::find_if(std::begin(info.m_OutputTensorInfos), std::end(info.m_OutputTensorInfos), checkType);
    if (it != std::end(info.m_OutputTensorInfos))
    {
        return true;
    }
    return false;
}

GpuFsaWorkloadFactory::GpuFsaWorkloadFactory(const std::shared_ptr<GpuFsaMemoryManager>& memoryManager)
    : m_MemoryManager(memoryManager)
{
    InitializeCLCompileContext();
}

GpuFsaWorkloadFactory::GpuFsaWorkloadFactory()
    : m_MemoryManager(new GpuFsaMemoryManager())
{
    InitializeCLCompileContext();
}

const BackendId& GpuFsaWorkloadFactory::GetBackendId() const
{
    return s_Id;
}

bool GpuFsaWorkloadFactory::IsLayerSupported(const Layer& layer,
                                             Optional<DataType> dataType,
                                             std::string& outReasonIfUnsupported)
{
    return IWorkloadFactory::IsLayerSupported(s_Id, layer, dataType, outReasonIfUnsupported);
}

std::unique_ptr<ITensorHandle> GpuFsaWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                         const bool /*isMemoryManaged*/) const
{
    std::unique_ptr<GpuFsaTensorHandle> tensorHandle = std::make_unique<GpuFsaTensorHandle>(tensorInfo);
    tensorHandle->SetMemoryGroup(m_MemoryManager->GetInterLayerMemoryGroup());

    return tensorHandle;
}

std::unique_ptr<ITensorHandle> GpuFsaWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                         DataLayout dataLayout,
                                                                         const bool /*isMemoryManaged*/) const
{
    std::unique_ptr<GpuFsaTensorHandle> tensorHandle = std::make_unique<GpuFsaTensorHandle>(tensorInfo, dataLayout);
    tensorHandle->SetMemoryGroup(m_MemoryManager->GetInterLayerMemoryGroup());

    return tensorHandle;
}


void GpuFsaWorkloadFactory::InitializeCLCompileContext() {
    // Initialize our m_CLCompileContext using default device and context
    auto context = arm_compute::CLKernelLibrary::get().context();
    auto device = arm_compute::CLKernelLibrary::get().get_device();
    m_CLCompileContext = arm_compute::CLCompileContext(context, device);
}

std::unique_ptr<IWorkload> GpuFsaWorkloadFactory::CreateWorkload(LayerType type,
                                                                 const QueueDescriptor& descriptor,
                                                                 const WorkloadInfo& info) const
{
    switch(type)
    {
        case LayerType::Constant :
        {
            auto constQueueDescriptor = PolymorphicDowncast<const ConstantQueueDescriptor*>(&descriptor);
            return std::make_unique<GpuFsaConstantWorkload>(*constQueueDescriptor, info, m_CLCompileContext);
        }
        case LayerType::Input :
        {
            auto inputQueueDescriptor = PolymorphicDowncast<const InputQueueDescriptor*>(&descriptor);
            return std::make_unique<CopyMemGenericWorkload>(*inputQueueDescriptor, info);
        }
        case LayerType::Output :
        {
            auto outputQueueDescriptor = PolymorphicDowncast<const OutputQueueDescriptor*>(&descriptor);
            return std::make_unique<CopyMemGenericWorkload>(*outputQueueDescriptor, info);
        }
        case LayerType::MemCopy :
        {
            auto memCopyQueueDescriptor = PolymorphicDowncast<const MemCopyQueueDescriptor*>(&descriptor);
            if (memCopyQueueDescriptor->m_Inputs.empty() || !memCopyQueueDescriptor->m_Inputs[0])
            {
                throw InvalidArgumentException("GpuFsaWorkloadFactory: Invalid null input for MemCopy workload");
            }
            return std::make_unique<CopyMemGenericWorkload>(*memCopyQueueDescriptor, info);
        }
        case LayerType::PreCompiled :
        {
            auto precompiledQueueDescriptor = PolymorphicDowncast<const PreCompiledQueueDescriptor*>(&descriptor);
            return std::make_unique<GpuFsaPreCompiledWorkload>(*precompiledQueueDescriptor, info);
        }
        default :
            return nullptr;
    }
}

} // namespace armnn