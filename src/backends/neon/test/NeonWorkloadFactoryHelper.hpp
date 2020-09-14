//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/IMemoryManager.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <backendsCommon/test/WorkloadFactoryHelper.hpp>

#include <neon/NeonBackend.hpp>
#include <neon/NeonWorkloadFactory.hpp>
#include "neon/NeonTensorHandleFactory.hpp"

namespace
{

template<>
struct WorkloadFactoryHelper<armnn::NeonWorkloadFactory>
{
    static armnn::IBackendInternal::IMemoryManagerSharedPtr GetMemoryManager()
    {
        armnn::NeonBackend backend;
        return backend.CreateMemoryManager();
    }

    static armnn::NeonWorkloadFactory GetFactory(
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ModelOptions& modelOptions = {})
    {
        armnn::NeonBackend backend;
        return armnn::NeonWorkloadFactory(armnn::PolymorphicPointerDowncast<armnn::NeonMemoryManager>(memoryManager),
                                          backend.CreateBackendSpecificModelContext(modelOptions));
    }

    static armnn::NeonTensorHandleFactory GetTensorHandleFactory(
            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr)
    {

        return armnn::NeonTensorHandleFactory(
            armnn::PolymorphicPointerDowncast<armnn::NeonMemoryManager>(memoryManager));
    }
};

using NeonWorkloadFactoryHelper = WorkloadFactoryHelper<armnn::NeonWorkloadFactory>;

} // anonymous namespace
