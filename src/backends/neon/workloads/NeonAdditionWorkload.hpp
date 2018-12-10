//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <neon/workloads/NeonWorkloadUtils.hpp>

namespace armnn
{

arm_compute::Status NeonAdditionWorkloadValidate(const TensorInfo& input0,
                                                 const TensorInfo& input1,
                                                 const TensorInfo& output);

class NeonAdditionWorkload : public BaseWorkload<AdditionQueueDescriptor>
{
public:
    NeonAdditionWorkload(const AdditionQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEArithmeticAddition m_AddLayer;
};

} //namespace armnn



