//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/test/WorkloadFactoryHelper.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <reference/RefBackend.hpp>
#include <reference/RefWorkloadFactory.hpp>
#include "reference/RefTensorHandleFactory.hpp"

namespace
{

template<>
struct WorkloadFactoryHelper<armnn::RefWorkloadFactory>
{
    static armnn::IBackendInternal::IMemoryManagerSharedPtr GetMemoryManager()
    {
        armnn::RefBackend backend;
        return backend.CreateMemoryManager();
    }

    static armnn::RefWorkloadFactory GetFactory(
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr)
    {
        IgnoreUnused(memoryManager);
        return armnn::RefWorkloadFactory();
    }

    static armnn::RefTensorHandleFactory GetTensorHandleFactory(
            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr)
    {

        return armnn::RefTensorHandleFactory(armnn::PolymorphicPointerDowncast<armnn::RefMemoryManager>(memoryManager));
    }
};

using RefWorkloadFactoryHelper = WorkloadFactoryHelper<armnn::RefWorkloadFactory>;

} // anonymous namespace
