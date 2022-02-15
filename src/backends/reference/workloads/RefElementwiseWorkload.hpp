//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>
#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>
#include "BaseIterator.hpp"
#include "ElementwiseFunction.hpp"
#include "Maximum.hpp"
#include "Minimum.hpp"
#include "StringMapping.hpp"

namespace armnn
{

template <typename Functor, typename ParentDescriptor, typename armnn::StringMapping::Id DebugString>
class RefElementwiseWorkload : public RefBaseWorkload<ParentDescriptor>
{
public:
    using InType = typename ElementwiseBinaryFunction<Functor>::InType;
    using OutType = typename ElementwiseBinaryFunction<Functor>::OutType;
    using RefBaseWorkload<ParentDescriptor>::m_Data;

    RefElementwiseWorkload(const ParentDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)  override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
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
