//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <Layer.hpp>
#include <armnn/backends/MemCopyWorkload.hpp>
#include <backendsCommon/MemImportWorkload.hpp>
#include <backendsCommon/MakeWorkloadHelper.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include "TosaRefWorkloadFactory.hpp"
#include "TosaRefBackendId.hpp"
#include "workloads/TosaRefWorkloads.hpp"
#include "TosaRefTensorHandle.hpp"
#include "TosaRefWorkloadFactory.hpp"


namespace armnn
{

namespace
{
static const BackendId s_Id{TosaRefBackendId()};
}
template <typename F32Workload, typename U8Workload, typename QueueDescriptorType>
std::unique_ptr<IWorkload> TosaRefWorkloadFactory::MakeWorkload(const QueueDescriptorType& descriptor,
                                                            const WorkloadInfo& info) const
{
    return MakeWorkloadHelper<NullWorkload, F32Workload, U8Workload, NullWorkload, NullWorkload, NullWorkload>
           (descriptor, info);
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

TosaRefWorkloadFactory::TosaRefWorkloadFactory(const std::shared_ptr<TosaRefMemoryManager>& memoryManager)
    : m_MemoryManager(memoryManager)
{
}

TosaRefWorkloadFactory::TosaRefWorkloadFactory()
    : m_MemoryManager(new TosaRefMemoryManager())
{
}

const BackendId& TosaRefWorkloadFactory::GetBackendId() const
{
    return s_Id;
}

bool TosaRefWorkloadFactory::IsLayerSupported(const Layer& layer,
                                              Optional<DataType> dataType,
                                              std::string& outReasonIfUnsupported)
{
    return IWorkloadFactory::IsLayerSupported(s_Id, layer, dataType, outReasonIfUnsupported);
}

bool TosaRefWorkloadFactory::IsLayerSupported(const IConnectableLayer& layer,
                                              Optional<DataType> dataType,
                                              std::string& outReasonIfUnsupported,
                                              const ModelOptions& modelOptions)
{
    return IWorkloadFactory::IsLayerSupported(s_Id, layer, dataType, outReasonIfUnsupported, modelOptions);
}

std::unique_ptr<ITensorHandle> TosaRefWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                          const bool isMemoryManaged) const
{
    if (isMemoryManaged)
    {
        return std::make_unique<TosaRefTensorHandle>(tensorInfo, m_MemoryManager);
    }
    else
    {
        return std::make_unique<TosaRefTensorHandle>(tensorInfo, static_cast<unsigned int>(MemorySource::Malloc));
    }
}

std::unique_ptr<ITensorHandle> TosaRefWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                          DataLayout dataLayout,
                                                                          const bool isMemoryManaged) const
{
    // For TosaRef it is okay to make the TensorHandle memory managed as it can also store a pointer
    // to unmanaged memory. This also ensures memory alignment.
    IgnoreUnused(isMemoryManaged, dataLayout);

    if (isMemoryManaged)
    {
        return std::make_unique<TosaRefTensorHandle>(tensorInfo, m_MemoryManager);
    }
    else
    {
        return std::make_unique<TosaRefTensorHandle>(tensorInfo, static_cast<unsigned int>(MemorySource::Malloc));
    }
}

std::unique_ptr<IWorkload> TosaRefWorkloadFactory::CreateWorkload(LayerType type,
                                                                  const QueueDescriptor& descriptor,
                                                                  const WorkloadInfo& info) const
{
    switch(type)
    {
        case LayerType::PreCompiled:
        {
            auto precompiledQueueDescriptor = PolymorphicDowncast<const PreCompiledQueueDescriptor*>(&descriptor);
            return std::make_unique<TosaRefPreCompiledWorkload>(*precompiledQueueDescriptor, info);
        }
        default:
            return nullptr;
    }
}

} // namespace armnn
