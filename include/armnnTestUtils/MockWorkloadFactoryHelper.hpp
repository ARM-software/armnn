//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "MockBackend.hpp"
#include "MockTensorHandleFactory.hpp"
#include <backendsCommon/test/WorkloadFactoryHelper.hpp>

namespace
{

template <>
struct WorkloadFactoryHelper<armnn::MockWorkloadFactory>
{
    static armnn::IBackendInternal::IMemoryManagerSharedPtr GetMemoryManager()
    {
        armnn::MockBackend backend;
        return backend.CreateMemoryManager();
    }

    static armnn::MockWorkloadFactory
        GetFactory(const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr)
    {
        IgnoreUnused(memoryManager);
        return armnn::MockWorkloadFactory();
    }

    static armnn::MockTensorHandleFactory
        GetTensorHandleFactory(const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr)
    {

        return armnn::MockTensorHandleFactory(std::static_pointer_cast<armnn::MockMemoryManager>(memoryManager));
    }
};

using MockWorkloadFactoryHelper = WorkloadFactoryHelper<armnn::MockWorkloadFactory>;

}    // anonymous namespace
