//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/neon/workloads/NeonWorkloadUtils.hpp>

namespace armnn
{

class NeonActivationUint8Workload : public Uint8Workload<ActivationQueueDescriptor>
{
public:
    NeonActivationUint8Workload(const ActivationQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEActivationLayer m_ActivationLayer;
};

} //namespace armnn





