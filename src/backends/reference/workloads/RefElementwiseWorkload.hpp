//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>
#include <backendsCommon/StringMapping.hpp>
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include "Maximum.hpp"

namespace armnn
{

template <typename Functor,
          typename armnn::DataType DataType,
          typename ParentDescriptor,
          typename armnn::StringMapping::Id DebugString>
class RefElementwiseWorkload
{
    // Needs specialization. The default is empty on purpose.
};

template <typename ParentDescriptor, typename Functor>
class BaseFloat32ElementwiseWorkload : public Float32Workload<ParentDescriptor>
{
public:
    using Float32Workload<ParentDescriptor>::Float32Workload;
    void ExecuteImpl(const char * debugString) const;
};

template <typename Functor,
          typename ParentDescriptor,
          typename armnn::StringMapping::Id DebugString>
class RefElementwiseWorkload<Functor, armnn::DataType::Float32, ParentDescriptor, DebugString>
    : public BaseFloat32ElementwiseWorkload<ParentDescriptor, Functor>
{
public:
    using BaseFloat32ElementwiseWorkload<ParentDescriptor, Functor>::BaseFloat32ElementwiseWorkload;

    virtual void Execute() const override
    {
        using Parent = BaseFloat32ElementwiseWorkload<ParentDescriptor, Functor>;
        Parent::ExecuteImpl(StringMapping::Instance().Get(DebugString));
    }
};

template <typename ParentDescriptor, typename Functor>
class BaseUint8ElementwiseWorkload : public Uint8Workload<ParentDescriptor>
{
public:
    using Uint8Workload<ParentDescriptor>::Uint8Workload;
    void ExecuteImpl(const char * debugString) const;
};

template <typename Functor,
          typename ParentDescriptor,
          typename armnn::StringMapping::Id DebugString>
class RefElementwiseWorkload<Functor, armnn::DataType::QuantisedAsymm8, ParentDescriptor, DebugString>
    : public BaseUint8ElementwiseWorkload<ParentDescriptor, Functor>
{
public:
    using BaseUint8ElementwiseWorkload<ParentDescriptor, Functor>::BaseUint8ElementwiseWorkload;

    virtual void Execute() const override
    {
        using Parent = BaseUint8ElementwiseWorkload<ParentDescriptor, Functor>;
        Parent::ExecuteImpl(StringMapping::Instance().Get(DebugString));
    }
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

} // armnn
