//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

namespace armnn
{

arm_compute::Status NeonSubtractionWorkloadValidate(const TensorInfo& input0,
                                                    const TensorInfo& input1,
                                                    const TensorInfo& output);

class NeonSubtractionFloatWorkload : public FloatWorkload<SubtractionQueueDescriptor>
{
public:
    NeonSubtractionFloatWorkload(const SubtractionQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEArithmeticSubtraction m_AddLayer;
};

} //namespace armnn
