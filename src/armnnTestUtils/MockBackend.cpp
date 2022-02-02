//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/backends/MemCopyWorkload.hpp>
#include <armnnTestUtils/MockBackend.hpp>
#include <armnnTestUtils/MockTensorHandle.hpp>

namespace armnn
{

constexpr const char* MockBackendId()
{
    return "CpuMock";
}

const BackendId& MockBackend::GetIdStatic()
{
    static const BackendId s_Id{MockBackendId()};
    return s_Id;
}

namespace
{
static const BackendId s_Id{ MockBackendId() };
}

MockWorkloadFactory::MockWorkloadFactory(const std::shared_ptr<MockMemoryManager>& memoryManager)
    : m_MemoryManager(memoryManager)
{}

MockWorkloadFactory::MockWorkloadFactory()
    : m_MemoryManager(new MockMemoryManager())
{}

const BackendId& MockWorkloadFactory::GetBackendId() const
{
    return s_Id;
}

std::unique_ptr<IWorkload> MockWorkloadFactory::CreateWorkload(LayerType type,
                                                               const QueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    switch (type)
    {
        case LayerType::MemCopy: {
            auto memCopyQueueDescriptor = PolymorphicDowncast<const MemCopyQueueDescriptor*>(&descriptor);
            if (descriptor.m_Inputs.empty())
            {
                throw InvalidArgumentException("MockWorkloadFactory: CreateMemCopy() expected an input tensor.");
            }
            return std::make_unique<CopyMemGenericWorkload>(*memCopyQueueDescriptor, info);
        }
        default:
            return nullptr;
    }
}

}    // namespace armnn