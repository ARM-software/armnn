//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/IMemoryManager.hpp>
#include <backendsCommon/test/WorkloadFactoryHelper.hpp>

#include <neon/NeonWorkloadFactory.hpp>

#include <arm_compute/runtime/Allocator.h>

#include <boost/polymorphic_pointer_cast.hpp>

namespace
{

template<>
struct WorkloadFactoryHelper<armnn::NeonWorkloadFactory>
{
    static armnn::NeonWorkloadFactory GetFactory()
    {
        armnn::IBackendInternal::IMemoryManagerSharedPtr memoryManager =
            std::make_shared<armnn::NeonMemoryManager>(std::make_unique<arm_compute::Allocator>(),
                                                       armnn::BaseMemoryManager::MemoryAffinity::Offset);

        return armnn::NeonWorkloadFactory(
            boost::polymorphic_pointer_downcast<armnn::NeonMemoryManager>(memoryManager));
    }
};

using NeonWorkloadFactoryHelper = WorkloadFactoryHelper<armnn::NeonWorkloadFactory>;

} // anonymous namespace
