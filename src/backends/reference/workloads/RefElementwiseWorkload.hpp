//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include "Maximum.hpp"
#include "Minimum.hpp"
#include "StringMapping.hpp"

namespace armnn
{

template <typename Functor, typename ParentDescriptor, typename armnn::StringMapping::Id DebugString>
class RefElementwiseWorkload : public BaseWorkload<ParentDescriptor>
{
public:
    using BaseWorkload<ParentDescriptor>::m_Data;
    using BaseWorkload<ParentDescriptor>::BaseWorkload;

    void Execute() const override;
};

using RefAdditionWorkload =
    RefElementwiseWorkload<std::plus<float>,
                          AdditionQueueDescriptor,
                          StringMapping::RefAdditionWorkload_Execute>;

using RefSubtractionWorkload =
    RefElementwiseWorkload<std::minus<float>,
                          SubtractionQueueDescriptor,
                          StringMapping::RefSubtractionWorkload_Execute>;

using RefMultiplicationWorkload =
    RefElementwiseWorkload<std::multiplies<float>,
                          MultiplicationQueueDescriptor,
                          StringMapping::RefMultiplicationWorkload_Execute>;

using RefDivisionWorkload =
    RefElementwiseWorkload<std::divides<float>,
                          DivisionQueueDescriptor,
                          StringMapping::RefDivisionWorkload_Execute>;

using RefMaximumWorkload =
    RefElementwiseWorkload<armnn::maximum<float>,
                          MaximumQueueDescriptor,
                          StringMapping::RefMaximumWorkload_Execute>;

using RefMinimumWorkload =
    RefElementwiseWorkload<armnn::minimum<float>,
                          MinimumQueueDescriptor,
                          StringMapping::RefMinimumWorkload_Execute>;
} // armnn
