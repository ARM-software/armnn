//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/test/WorkloadFactoryHelper.hpp>

#include <reference/RefBackend.hpp>
#include <reference/RefWorkloadFactory.hpp>

namespace
{

template<>
struct WorkloadFactoryHelper<armnn::RefWorkloadFactory>
{
    static armnn::RefWorkloadFactory GetFactory()
    {
        return armnn::RefWorkloadFactory();
    }
};

using RefWorkloadFactoryHelper = WorkloadFactoryHelper<armnn::RefWorkloadFactory>;

} // anonymous namespace
