//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/backends/MemCopyWorkload.hpp>
#include <armnn/backends/TensorHandle.hpp>

#include "SampleDynamicAdditionWorkload.hpp"
#include "SampleDynamicBackend.hpp"
#include "SampleDynamicWorkloadFactory.hpp"
#include "SampleTensorHandle.hpp"

namespace sdb // sample dynamic backend
{

namespace
{
static const armnn::BackendId s_Id{  GetBackendId() };
}

SampleDynamicWorkloadFactory::SampleDynamicWorkloadFactory(const std::shared_ptr<SampleMemoryManager>& memoryManager)
    : m_MemoryManager(memoryManager)
{
}

SampleDynamicWorkloadFactory::SampleDynamicWorkloadFactory()
    : m_MemoryManager(new SampleMemoryManager())
{
}

const armnn::BackendId& SampleDynamicWorkloadFactory::GetBackendId() const
{
    return s_Id;
}

bool SampleDynamicWorkloadFactory::IsLayerSupported(const armnn::IConnectableLayer& layer,
                                                    armnn::Optional<armnn::DataType> dataType,
                                                    std::string& outReasonIfUnsupported)
{
    return IWorkloadFactory::IsLayerSupported(s_Id, layer, dataType, outReasonIfUnsupported);
}

std::unique_ptr<armnn::ITensorHandle> SampleDynamicWorkloadFactory::CreateTensorHandle(
        const armnn::TensorInfo& tensorInfo,
        const bool isMemoryManaged) const
{
    return std::make_unique<armnn::ScopedTensorHandle>(tensorInfo);
}

std::unique_ptr<armnn::ITensorHandle> SampleDynamicWorkloadFactory::CreateTensorHandle(
        const armnn::TensorInfo& tensorInfo,
        armnn::DataLayout dataLayout,
        const bool isMemoryManaged) const
{
    return std::make_unique<armnn::ScopedTensorHandle>(tensorInfo);
}

std::unique_ptr<armnn::IWorkload> SampleDynamicWorkloadFactory::CreateAddition(
        const armnn::AdditionQueueDescriptor& descriptor,
        const armnn::WorkloadInfo& info) const
{
    return std::make_unique<SampleDynamicAdditionWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> SampleDynamicWorkloadFactory::CreateInput(
        const armnn::InputQueueDescriptor& descriptor,
        const armnn::WorkloadInfo& info) const
{
    return std::make_unique<armnn::CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> SampleDynamicWorkloadFactory::CreateOutput(
        const armnn::OutputQueueDescriptor& descriptor,
        const armnn::WorkloadInfo& info) const
{
    return std::make_unique<armnn::CopyMemGenericWorkload>(descriptor, info);
}

} // namespace sdb
