//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/test/WorkloadFactoryHelper.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <gpuFsa/GpuFsaBackend.hpp>
#include <gpuFsa/GpuFsaWorkloadFactory.hpp>
#include "gpuFsa/GpuFsaTensorHandleFactory.hpp"

namespace
{

template<>
struct WorkloadFactoryHelper<armnn::GpuFsaWorkloadFactory>
{
    static armnn::IBackendInternal::IMemoryManagerSharedPtr GetMemoryManager()
    {
        armnn::GpuFsaBackend backend;
        return backend.CreateMemoryManager();
    }

    static armnn::GpuFsaWorkloadFactory GetFactory(
        const armnn::IBackendInternal::IMemoryManagerSharedPtr&)
    {
        return armnn::GpuFsaWorkloadFactory();
    }

    static armnn::GpuFsaTensorHandleFactory GetTensorHandleFactory(
            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr)
    {

        return armnn::GpuFsaTensorHandleFactory(
                armnn::PolymorphicPointerDowncast<armnn::GpuFsaMemoryManager>(memoryManager));
    }
};

using GpuFsaWorkloadFactoryHelper = WorkloadFactoryHelper<armnn::GpuFsaWorkloadFactory>;

} // anonymous namespace
