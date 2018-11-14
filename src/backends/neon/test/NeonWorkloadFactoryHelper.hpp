//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/IMemoryManager.hpp>
#include <backendsCommon/test/WorkloadFactoryHelper.hpp>

#include <neon/NeonBackend.hpp>
#include <neon/NeonWorkloadFactory.hpp>

#include <boost/polymorphic_pointer_cast.hpp>

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
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
    {
        return armnn::NeonWorkloadFactory(
            boost::polymorphic_pointer_downcast<armnn::NeonMemoryManager>(memoryManager));
    }
};

using NeonWorkloadFactoryHelper = WorkloadFactoryHelper<armnn::NeonWorkloadFactory>;

} // anonymous namespace
