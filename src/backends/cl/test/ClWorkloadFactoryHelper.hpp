//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/IMemoryManager.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <backendsCommon/test/WorkloadFactoryHelper.hpp>

#include <cl/ClBackend.hpp>
#include <cl/ClWorkloadFactory.hpp>
#include <cl/ClTensorHandleFactory.hpp>

namespace
{

template<>
struct WorkloadFactoryHelper<armnn::ClWorkloadFactory>
{
    static armnn::IBackendInternal::IMemoryManagerSharedPtr GetMemoryManager()
    {
        armnn::ClBackend backend;
        return backend.CreateMemoryManager();
    }

    static armnn::ClWorkloadFactory GetFactory(
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ModelOptions& modelOptions = {})
    {
        armnn::ClBackend backend;
        return armnn::ClWorkloadFactory(armnn::PolymorphicPointerDowncast<armnn::ClMemoryManager>(memoryManager),
                                        backend.CreateBackendSpecificModelContext(modelOptions));
    }

    static armnn::ClTensorHandleFactory GetTensorHandleFactory(
            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr)
    {

        return armnn::ClTensorHandleFactory(armnn::PolymorphicPointerDowncast<armnn::ClMemoryManager>(memoryManager));
    }
};

using ClWorkloadFactoryHelper = WorkloadFactoryHelper<armnn::ClWorkloadFactory>;

} // anonymous namespace
