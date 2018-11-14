//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/IMemoryManager.hpp>
#include <backendsCommon/test/WorkloadFactoryHelper.hpp>

#include <cl/ClBackend.hpp>
#include <cl/ClWorkloadFactory.hpp>

#include <boost/polymorphic_pointer_cast.hpp>

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
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
    {
        return armnn::ClWorkloadFactory(boost::polymorphic_pointer_downcast<armnn::ClMemoryManager>(memoryManager));
    }
};

using ClWorkloadFactoryHelper = WorkloadFactoryHelper<armnn::ClWorkloadFactory>;

} // anonymous namespace
