//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/CLFunctions.h>

namespace armnn
{

arm_compute::Status ClGreaterWorkloadValidate(const TensorInfo& input0,
                                              const TensorInfo& input1,
                                              const TensorInfo& output);

template<DataType T>
class ClGreaterWorkload : public MultiTypedWorkload<GreaterQueueDescriptor, T, DataType::Boolean>
{
public:
    ClGreaterWorkload(const GreaterQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    using MultiTypedWorkload<GreaterQueueDescriptor, T, DataType::Boolean>::m_Data;
    mutable arm_compute::CLComparison m_GreaterLayer;
};

using ClGreaterFloat32Workload = ClGreaterWorkload<DataType::Float32>;
using ClGreaterUint8Workload = ClGreaterWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn
