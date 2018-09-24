//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/neon/workloads/NeonWorkloadUtils.hpp>

namespace armnn
{

arm_compute::Status NeonAdditionWorkloadValidate(const TensorInfo& input0,
                                                 const TensorInfo& input1,
                                                 const TensorInfo& output);

class NeonAdditionFloatWorkload : public FloatWorkload<AdditionQueueDescriptor>
{
public:
    NeonAdditionFloatWorkload(const AdditionQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEArithmeticAddition m_AddLayer;
};

} //namespace armnn



