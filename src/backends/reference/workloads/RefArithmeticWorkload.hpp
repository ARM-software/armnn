//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>
#include <backends/StringMapping.hpp>
#include <backends/Workload.hpp>
#include <backends/WorkloadData.hpp>

namespace armnn
{

template <typename Functor,
          typename armnn::DataType DataType,
          typename ParentDescriptor,
          typename armnn::StringMapping::Id DebugString>
class RefArithmeticWorkload
{
    // Needs specialization. The default is empty on purpose.
};

template <typename ParentDescriptor, typename Functor>
class BaseFloat32ArithmeticWorkload : public Float32Workload<ParentDescriptor>
{
public:
    using Float32Workload<ParentDescriptor>::Float32Workload;
    void ExecuteImpl(const char * debugString) const;
};

template <typename Functor,
          typename ParentDescriptor,
          typename armnn::StringMapping::Id DebugString>
class RefArithmeticWorkload<Functor, armnn::DataType::Float32, ParentDescriptor, DebugString>
    : public BaseFloat32ArithmeticWorkload<ParentDescriptor, Functor>
{
public:
    using BaseFloat32ArithmeticWorkload<ParentDescriptor, Functor>::BaseFloat32ArithmeticWorkload;

    virtual void Execute() const override
    {
        using Parent = BaseFloat32ArithmeticWorkload<ParentDescriptor, Functor>;
        Parent::ExecuteImpl(StringMapping::Instance().Get(DebugString));
    }
};

template <typename ParentDescriptor, typename Functor>
class BaseUint8ArithmeticWorkload : public Uint8Workload<ParentDescriptor>
{
public:
    using Uint8Workload<ParentDescriptor>::Uint8Workload;
    void ExecuteImpl(const char * debugString) const;
};

template <typename Functor,
          typename ParentDescriptor,
          typename armnn::StringMapping::Id DebugString>
class RefArithmeticWorkload<Functor, armnn::DataType::QuantisedAsymm8, ParentDescriptor, DebugString>
    : public BaseUint8ArithmeticWorkload<ParentDescriptor, Functor>
{
public:
    using BaseUint8ArithmeticWorkload<ParentDescriptor, Functor>::BaseUint8ArithmeticWorkload;

    virtual void Execute() const override
    {
        using Parent = BaseUint8ArithmeticWorkload<ParentDescriptor, Functor>;
        Parent::ExecuteImpl(StringMapping::Instance().Get(DebugString));
    }
};

using RefAdditionFloat32Workload =
    RefArithmeticWorkload<std::plus<float>,
                          DataType::Float32,
                          AdditionQueueDescriptor,
                          StringMapping::RefAdditionWorkload_Execute>;

using RefAdditionUint8Workload =
    RefArithmeticWorkload<std::plus<float>,
                          DataType::QuantisedAsymm8,
                          AdditionQueueDescriptor,
                          StringMapping::RefAdditionWorkload_Execute>;


using RefSubtractionFloat32Workload =
    RefArithmeticWorkload<std::minus<float>,
                          DataType::Float32,
                          SubtractionQueueDescriptor,
                          StringMapping::RefSubtractionWorkload_Execute>;

using RefSubtractionUint8Workload =
    RefArithmeticWorkload<std::minus<float>,
                          DataType::QuantisedAsymm8,
                          SubtractionQueueDescriptor,
                          StringMapping::RefSubtractionWorkload_Execute>;

using RefMultiplicationFloat32Workload =
    RefArithmeticWorkload<std::multiplies<float>,
                          DataType::Float32,
                          MultiplicationQueueDescriptor,
                          StringMapping::RefMultiplicationWorkload_Execute>;

using RefMultiplicationUint8Workload =
    RefArithmeticWorkload<std::multiplies<float>,
                          DataType::QuantisedAsymm8,
                          MultiplicationQueueDescriptor,
                          StringMapping::RefMultiplicationWorkload_Execute>;

using RefDivisionFloat32Workload =
    RefArithmeticWorkload<std::divides<float>,
                          DataType::Float32,
                          DivisionQueueDescriptor,
                          StringMapping::RefDivisionWorkload_Execute>;

using RefDivisionUint8Workload =
    RefArithmeticWorkload<std::divides<float>,
                          DataType::QuantisedAsymm8,
                          DivisionQueueDescriptor,
                          StringMapping::RefDivisionWorkload_Execute>;

} // armnn
