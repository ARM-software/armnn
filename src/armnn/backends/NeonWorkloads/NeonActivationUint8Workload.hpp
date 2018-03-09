//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

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





