//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

// Activation layer execution.
class ClActivationUint8Workload : public Uint8Workload<ActivationQueueDescriptor>
{
public:
    ClActivationUint8Workload(const ActivationQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::CLActivationLayer m_ActivationLayer;
};

} //namespace armnn



