//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/test/WorkloadFactoryHelper.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <tosaReference/TosaRefBackend.hpp>
#include <tosaReference/TosaRefWorkloadFactory.hpp>
#include "tosaReference/TosaRefTensorHandleFactory.hpp"

namespace
{

template<>
struct WorkloadFactoryHelper<armnn::TosaRefWorkloadFactory>
{
    static armnn::IBackendInternal::IMemoryManagerSharedPtr GetMemoryManager()
    {
        armnn::TosaRefBackend backend;
        return backend.CreateMemoryManager();
    }

    static armnn::TosaRefWorkloadFactory GetFactory(
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr)
    {
        IgnoreUnused(memoryManager);
        return armnn::TosaRefWorkloadFactory();
    }

    static armnn::TosaRefTensorHandleFactory GetTensorHandleFactory(
            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr)
    {

        return armnn::TosaRefTensorHandleFactory(
                armnn::PolymorphicPointerDowncast<armnn::TosaRefMemoryManager>(memoryManager));
    }
};

using TosaRefWorkloadFactoryHelper = WorkloadFactoryHelper<armnn::TosaRefWorkloadFactory>;

} // anonymous namespace
