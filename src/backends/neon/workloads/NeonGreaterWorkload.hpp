//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <neon/workloads/NeonWorkloadUtils.hpp>

#include <arm_compute/runtime/NEON/functions/NEElementwiseOperations.h>

namespace armnn
{

arm_compute::Status NeonGreaterWorkloadValidate(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output);

template <DataType T>
class NeonGreaterWorkload : public MultiTypedWorkload<GreaterQueueDescriptor, T, DataType::Boolean>
{
public:
    using MultiTypedWorkload<GreaterQueueDescriptor, T, DataType::Boolean>::m_Data;

    NeonGreaterWorkload(const GreaterQueueDescriptor& descriptor, const WorkloadInfo& info);

    virtual void Execute() const override;

private:
    mutable arm_compute::NEGreater m_GreaterLayer;
};

using NeonGreaterFloat32Workload = NeonGreaterWorkload<DataType::Float32>;
using NeonGreaterUint8Workload = NeonGreaterWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn