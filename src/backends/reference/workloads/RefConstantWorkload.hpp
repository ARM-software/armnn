//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

#include <armnn/Types.hpp>

namespace armnn
{

// Base class template providing an implementation of the Constant layer common to all data types.
template <armnn::DataType DataType>
class RefConstantWorkload : public TypedWorkload<ConstantQueueDescriptor, DataType>
{
public:
    RefConstantWorkload(const ConstantQueueDescriptor& descriptor, const WorkloadInfo& info)
        : TypedWorkload<ConstantQueueDescriptor, DataType>(descriptor, info)
        , m_RanOnce(false)
    {
    }

    using TypedWorkload<ConstantQueueDescriptor, DataType>::m_Data;
    using TypedWorkload<ConstantQueueDescriptor, DataType>::TypedWorkload;

    virtual void Execute() const override;

private:
    mutable bool m_RanOnce;
};

using RefConstantFloat32Workload = RefConstantWorkload<DataType::Float32>;
using RefConstantUint8Workload = RefConstantWorkload<DataType::QuantisedAsymm8>;
using RefConstantInt32Workload = RefConstantWorkload<DataType::Signed32>;

} //namespace armnn
