//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include "BaseIterator.hpp"
#include "ElementwiseFunction.hpp"
#include "Maximum.hpp"
#include "Minimum.hpp"
#include "StringMapping.hpp"

namespace armnn
{

template <typename Functor, typename ParentDescriptor, typename armnn::StringMapping::Id DebugString>
class RefElementwiseWorkload : public BaseWorkload<ParentDescriptor>
{
public:
    using InType = typename ElementwiseFunction<Functor>::InType;
    using OutType = typename ElementwiseFunction<Functor>::OutType;
    using BaseWorkload<ParentDescriptor>::m_Data;

    RefElementwiseWorkload(const ParentDescriptor& descriptor, const WorkloadInfo& info);
    void PostAllocationConfigure() override;
    void Execute() const override;

private:
    std::unique_ptr<Decoder<InType>> m_Input0;
    std::unique_ptr<Decoder<InType>> m_Input1;
    std::unique_ptr<Encoder<OutType>> m_Output;
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

using RefEqualWorkload =
    RefElementwiseWorkload<std::equal_to<float>,
                           armnn::EqualQueueDescriptor,
                           armnn::StringMapping::RefEqualWorkload_Execute>;

using RefGreaterWorkload =
    RefElementwiseWorkload<std::greater<float>,
                           armnn::GreaterQueueDescriptor,
                           armnn::StringMapping::RefGreaterWorkload_Execute>;
} // armnn
