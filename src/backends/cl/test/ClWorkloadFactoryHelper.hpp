//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/IMemoryManager.hpp>
#include <backendsCommon/test/WorkloadFactoryHelper.hpp>

#include <cl/ClWorkloadFactory.hpp>

#include <arm_compute/runtime/CL/CLBufferAllocator.h>

#include <boost/polymorphic_pointer_cast.hpp>

namespace
{

template<>
struct WorkloadFactoryHelper<armnn::ClWorkloadFactory>
{
    static armnn::ClWorkloadFactory GetFactory()
    {
        armnn::IBackendInternal::IMemoryManagerSharedPtr memoryManager =
            std::make_shared<armnn::ClMemoryManager>(std::make_unique<arm_compute::CLBufferAllocator>());

        return armnn::ClWorkloadFactory(boost::polymorphic_pointer_downcast<armnn::ClMemoryManager>(memoryManager));
    }
};

using ClWorkloadFactoryHelper = WorkloadFactoryHelper<armnn::ClWorkloadFactory>;

} // anonymous namespace
