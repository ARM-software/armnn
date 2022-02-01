//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/MemCopyWorkload.hpp>
#include <armnnTestUtils/MockTensorHandle.hpp>

namespace armnn
{

// A bare bones Mock backend to enable unit testing of simple tensor manipulation features.
class MockBackend : public IBackendInternal
{
public:
    MockBackend() = default;

    ~MockBackend() = default;

    static const BackendId& GetIdStatic();

    const BackendId& GetId() const override
    {
        return GetIdStatic();
    }
    IBackendInternal::IWorkloadFactoryPtr
        CreateWorkloadFactory(const IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr) const override
    {
        IgnoreUnused(memoryManager);
        return nullptr;
    }

    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override
    {
        return nullptr;
    };
};

class MockWorkloadFactory : public IWorkloadFactory
{

public:
    explicit MockWorkloadFactory(const std::shared_ptr<MockMemoryManager>& memoryManager);
    MockWorkloadFactory();

    ~MockWorkloadFactory()
    {}

    const BackendId& GetBackendId() const override;

    bool SupportsSubTensors() const override
    {
        return false;
    }

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateSubTensorHandle instead")
    std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle&,
                                                         TensorShape const&,
                                                         unsigned int const*) const override
    {
        return nullptr;
    }

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateTensorHandle instead")
    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      const bool IsMemoryManaged = true) const override
    {
        IgnoreUnused(IsMemoryManaged);
        return std::make_unique<MockTensorHandle>(tensorInfo, m_MemoryManager);
    };

    ARMNN_DEPRECATED_MSG("Use ITensorHandleFactory::CreateTensorHandle instead")
    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      DataLayout dataLayout,
                                                      const bool IsMemoryManaged = true) const override
    {
        IgnoreUnused(dataLayout, IsMemoryManaged);
        return std::make_unique<MockTensorHandle>(tensorInfo, static_cast<unsigned int>(MemorySource::Malloc));
    };

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE(
        "Use ABI stable "
        "CreateWorkload(LayerType, const QueueDescriptor&, const WorkloadInfo& info) instead.",
        "22.11")
    std::unique_ptr<IWorkload> CreateInput(const InputQueueDescriptor& descriptor,
                                           const WorkloadInfo& info) const override
    {
        if (info.m_InputTensorInfos.empty())
        {
            throw InvalidArgumentException("MockWorkloadFactory::CreateInput: Input cannot be zero length");
        }
        if (info.m_OutputTensorInfos.empty())
        {
            throw InvalidArgumentException("MockWorkloadFactory::CreateInput: Output cannot be zero length");
        }

        if (info.m_InputTensorInfos[0].GetNumBytes() != info.m_OutputTensorInfos[0].GetNumBytes())
        {
            throw InvalidArgumentException(
                "MockWorkloadFactory::CreateInput: data input and output differ in byte count.");
        }

        return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
    };

    std::unique_ptr<IWorkload>
        CreateWorkload(LayerType type, const QueueDescriptor& descriptor, const WorkloadInfo& info) const override;

private:
    mutable std::shared_ptr<MockMemoryManager> m_MemoryManager;
};

}    // namespace armnn
