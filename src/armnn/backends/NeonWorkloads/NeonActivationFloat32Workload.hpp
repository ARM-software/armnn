//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

namespace armnn
{
class NeonActivationFloat32Workload : public Float32Workload<ActivationQueueDescriptor>
{
public:
    NeonActivationFloat32Workload(const ActivationQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::NEActivationLayer m_ActivationLayer;
};
} //namespace armnn



