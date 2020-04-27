//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <neon/workloads/NeonWorkloadUtils.hpp>

namespace armnn
{
arm_compute::Status NeonConstantWorkloadValidate(const TensorInfo& output);

class NeonConstantWorkload : public BaseWorkload<ConstantQueueDescriptor>
{
public:
    NeonConstantWorkload(const ConstantQueueDescriptor& descriptor, const WorkloadInfo& info);

    virtual void Execute() const override;

private:
    mutable bool m_RanOnce;
};

} //namespace armnn
