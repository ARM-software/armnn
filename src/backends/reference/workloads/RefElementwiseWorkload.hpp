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

template <typename Functor,
          typename armnn::DataType DataType,
          typename ParentDescriptor,
          typename armnn::StringMapping::Id DebugString>
class RefElementwiseWorkload
    : public TypedWorkload<ParentDescriptor, DataType>
{
public:

    using TypedWorkload<ParentDescriptor, DataType>::m_Data;
    using TypedWorkload<ParentDescriptor, DataType>::TypedWorkload;

    void Execute() const override;
};

using RefAdditionFloat32Workload =
    RefElementwiseWorkload<std::plus<float>,
                          DataType::Float32,
                          AdditionQueueDescriptor,
                          StringMapping::RefAdditionWorkload_Execute>;

using RefAdditionUint8Workload =
    RefElementwiseWorkload<std::plus<float>,
                          DataType::QuantisedAsymm8,
                          AdditionQueueDescriptor,
                          StringMapping::RefAdditionWorkload_Execute>;

using RefSubtractionFloat32Workload =
    RefElementwiseWorkload<std::minus<float>,
                          DataType::Float32,
                          SubtractionQueueDescriptor,
                          StringMapping::RefSubtractionWorkload_Execute>;

using RefSubtractionUint8Workload =
    RefElementwiseWorkload<std::minus<float>,
                          DataType::QuantisedAsymm8,
                          SubtractionQueueDescriptor,
                          StringMapping::RefSubtractionWorkload_Execute>;

using RefMultiplicationFloat32Workload =
    RefElementwiseWorkload<std::multiplies<float>,
                          DataType::Float32,
                          MultiplicationQueueDescriptor,
                          StringMapping::RefMultiplicationWorkload_Execute>;

using RefMultiplicationUint8Workload =
    RefElementwiseWorkload<std::multiplies<float>,
                          DataType::QuantisedAsymm8,
                          MultiplicationQueueDescriptor,
                          StringMapping::RefMultiplicationWorkload_Execute>;

using RefDivisionFloat32Workload =
    RefElementwiseWorkload<std::divides<float>,
                          DataType::Float32,
                          DivisionQueueDescriptor,
                          StringMapping::RefDivisionWorkload_Execute>;

using RefDivisionUint8Workload =
    RefElementwiseWorkload<std::divides<float>,
                          DataType::QuantisedAsymm8,
                          DivisionQueueDescriptor,
                          StringMapping::RefDivisionWorkload_Execute>;

using RefMaximumFloat32Workload =
    RefElementwiseWorkload<armnn::maximum<float>,
                          DataType::Float32,
                          MaximumQueueDescriptor,
                          StringMapping::RefMaximumWorkload_Execute>;

using RefMaximumUint8Workload =
    RefElementwiseWorkload<armnn::maximum<float>,
                          DataType::QuantisedAsymm8,
                          MaximumQueueDescriptor,
                          StringMapping::RefMaximumWorkload_Execute>;

using RefMinimumFloat32Workload =
    RefElementwiseWorkload<minimum<float>,
                          DataType::Float32,
                          MinimumQueueDescriptor,
                          StringMapping::RefMinimumWorkload_Execute>;

using RefMinimumUint8Workload =
    RefElementwiseWorkload<minimum<float>,
                          DataType::QuantisedAsymm8,
                          MinimumQueueDescriptor,
                          StringMapping::RefMinimumWorkload_Execute>;
} // armnn
