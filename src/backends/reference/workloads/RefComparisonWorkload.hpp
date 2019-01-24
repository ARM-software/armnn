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

template <typename Functor,
          typename armnn::DataType DataType,
          typename ParentDescriptor,
          typename armnn::StringMapping::Id DebugString>
class RefComparisonWorkload
{
    // Needs specialization. The default is empty on purpose.
};

template <typename ParentDescriptor, typename Functor>
class RefFloat32ComparisonWorkload : public BaseFloat32ComparisonWorkload<ParentDescriptor>
{
public:
    using BaseFloat32ComparisonWorkload<ParentDescriptor>::BaseFloat32ComparisonWorkload;
    void ExecuteImpl(const char * debugString) const;
};

template <typename Functor, typename ParentDescriptor, typename armnn::StringMapping::Id DebugString>
class RefComparisonWorkload<Functor, armnn::DataType::Float32, ParentDescriptor, DebugString>
    : public RefFloat32ComparisonWorkload<ParentDescriptor, Functor>
{
public:
    using RefFloat32ComparisonWorkload<ParentDescriptor, Functor>::RefFloat32ComparisonWorkload;

    virtual void Execute() const override
    {
        using Parent = RefFloat32ComparisonWorkload<ParentDescriptor, Functor>;
        Parent::ExecuteImpl(StringMapping::Instance().Get(DebugString));
    }
};

template <typename ParentDescriptor, typename Functor>
class RefUint8ComparisonWorkload : public BaseUint8ComparisonWorkload<ParentDescriptor>
{
public:
    using BaseUint8ComparisonWorkload<ParentDescriptor>::BaseUint8ComparisonWorkload;
    void ExecuteImpl(const char * debugString) const;
};

template <typename Functor, typename ParentDescriptor, typename armnn::StringMapping::Id DebugString>
class RefComparisonWorkload<Functor, armnn::DataType::QuantisedAsymm8, ParentDescriptor, DebugString>
    : public RefUint8ComparisonWorkload<ParentDescriptor, Functor>
{
public:
    using RefUint8ComparisonWorkload<ParentDescriptor, Functor>::RefUint8ComparisonWorkload;

    virtual void Execute() const override
    {
        using Parent = RefUint8ComparisonWorkload<ParentDescriptor, Functor>;
        Parent::ExecuteImpl(StringMapping::Instance().Get(DebugString));
    }
};

using RefEqualFloat32Workload =
    RefComparisonWorkload<std::equal_to<float>,
                          DataType::Float32,
                          EqualQueueDescriptor,
                          StringMapping::RefEqualWorkload_Execute>;

using RefEqualUint8Workload =
    RefComparisonWorkload<std::equal_to<uint8_t>,
                          DataType::QuantisedAsymm8,
                          EqualQueueDescriptor,
                          StringMapping::RefEqualWorkload_Execute>;

using RefGreaterFloat32Workload =
    RefComparisonWorkload<std::greater<float>,
                          DataType::Float32,
                          GreaterQueueDescriptor,
                          StringMapping::RefGreaterWorkload_Execute>;

using RefGreaterUint8Workload =
    RefComparisonWorkload<std::greater<uint8_t>,
                          DataType::QuantisedAsymm8,
                          GreaterQueueDescriptor,
                          StringMapping::RefGreaterWorkload_Execute>;
} // armnn
