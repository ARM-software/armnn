//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include "StringMapping.hpp"

namespace armnn
{

template <typename Functor, typename ParentDescriptor, typename armnn::StringMapping::Id DebugString>
class RefComparisonWorkload : public BaseWorkload<ParentDescriptor>
{
public:
    using BaseWorkload<ParentDescriptor>::m_Data;
    using BaseWorkload<ParentDescriptor>::BaseWorkload;

    void Execute() const override;
};

using RefEqualWorkload =
    RefComparisonWorkload<std::equal_to<float>,
                         EqualQueueDescriptor,
                         StringMapping::RefEqualWorkload_Execute>;


using RefGreaterWorkload =
    RefComparisonWorkload<std::greater<float>,
                         GreaterQueueDescriptor,
                         StringMapping::RefGreaterWorkload_Execute>;
} // armnn
