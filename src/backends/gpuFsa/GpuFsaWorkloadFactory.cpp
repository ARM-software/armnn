//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <Layer.hpp>

#include "GpuFsaWorkloadFactory.hpp"
#include "GpuFsaBackendId.hpp"
#include "GpuFsaTensorHandle.hpp"

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
}

GpuFsaWorkloadFactory::GpuFsaWorkloadFactory()
    : m_MemoryManager(new GpuFsaMemoryManager())
{
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

std::unique_ptr<IWorkload> GpuFsaWorkloadFactory::CreateWorkload(LayerType /*type*/,
                                                                 const QueueDescriptor& /*descriptor*/,
                                                                 const WorkloadInfo& /*info*/) const
{
    return nullptr;
}

} // namespace armnn