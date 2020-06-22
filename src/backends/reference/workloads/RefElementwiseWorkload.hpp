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
    using InType = typename ElementwiseBinaryFunction<Functor>::InType;
    using OutType = typename ElementwiseBinaryFunction<Functor>::OutType;
    using BaseWorkload<ParentDescriptor>::m_Data;

    RefElementwiseWorkload(const ParentDescriptor& descriptor, const WorkloadInfo& info);
    void PostAllocationConfigure() override;
    void Execute() const override;

private:
    std::unique_ptr<Decoder<InType>> m_Input0;
    std::unique_ptr<Decoder<InType>> m_Input1;
    std::unique_ptr<Encoder<OutType>> m_Output;
};

template <typename DataType = float>
using RefAdditionWorkload =
    RefElementwiseWorkload<std::plus<DataType>,
                          AdditionQueueDescriptor,
                          StringMapping::RefAdditionWorkload_Execute>;

template <typename DataType = float>
using RefSubtractionWorkload =
    RefElementwiseWorkload<std::minus<DataType>,
                          SubtractionQueueDescriptor,
                          StringMapping::RefSubtractionWorkload_Execute>;

template <typename DataType = float>
using RefMultiplicationWorkload =
    RefElementwiseWorkload<std::multiplies<DataType>,
                          MultiplicationQueueDescriptor,
                          StringMapping::RefMultiplicationWorkload_Execute>;

template <typename DataType = float>
using RefDivisionWorkload =
    RefElementwiseWorkload<std::divides<DataType>,
                          DivisionQueueDescriptor,
                          StringMapping::RefDivisionWorkload_Execute>;

template <typename DataType = float>
using RefMaximumWorkload =
    RefElementwiseWorkload<armnn::maximum<DataType>,
                          MaximumQueueDescriptor,
                          StringMapping::RefMaximumWorkload_Execute>;

template <typename DataType = float>
using RefMinimumWorkload =
    RefElementwiseWorkload<armnn::minimum<DataType>,
                          MinimumQueueDescriptor,
                          StringMapping::RefMinimumWorkload_Execute>;

} // armnn
