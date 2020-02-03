//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/MemCopyWorkload.hpp>

#include "SampleDynamicAdditionWorkload.hpp"
#include "SampleDynamicBackend.hpp"
#include "SampleDynamicWorkloadFactory.hpp"
#include "SampleTensorHandle.hpp"

namespace armnn
{

namespace
{
static const BackendId s_Id{  GetBackendId() };
}

SampleDynamicWorkloadFactory::SampleDynamicWorkloadFactory(const std::shared_ptr<SampleMemoryManager>& memoryManager)
    : m_MemoryManager(memoryManager)
{
}

SampleDynamicWorkloadFactory::SampleDynamicWorkloadFactory()
    : m_MemoryManager(new SampleMemoryManager())
{
}

const BackendId& SampleDynamicWorkloadFactory::GetBackendId() const
{
    return s_Id;
}

bool SampleDynamicWorkloadFactory::IsLayerSupported(const IConnectableLayer& layer,
                                                    Optional<DataType> dataType,
                                                    std::string& outReasonIfUnsupported)
{
    return IWorkloadFactory::IsLayerSupported(s_Id, layer, dataType, outReasonIfUnsupported);
}

std::unique_ptr<ITensorHandle> SampleDynamicWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                                const bool isMemoryManaged) const
{
    return std::make_unique<ScopedCpuTensorHandle>(tensorInfo);
}

std::unique_ptr<ITensorHandle> SampleDynamicWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                                DataLayout dataLayout,
                                                                                const bool isMemoryManaged) const
{
    return std::make_unique<ScopedCpuTensorHandle>(tensorInfo);
}

std::unique_ptr<IWorkload> SampleDynamicWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                                        const WorkloadInfo& info) const
{
    return std::make_unique<SampleDynamicAdditionWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> SampleDynamicWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                                     const WorkloadInfo& info) const
{
    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> SampleDynamicWorkloadFactory::CreateOutput(const OutputQueueDescriptor& descriptor,
                                                                      const WorkloadInfo& info) const
{
    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

} // namespace armnn
