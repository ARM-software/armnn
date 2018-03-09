//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{
// Softmax
class ClSoftmaxUint8Workload : public Uint8Workload<SoftmaxQueueDescriptor>
{
public:
    ClSoftmaxUint8Workload(const SoftmaxQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;
private:

    mutable arm_compute::CLSoftmaxLayer m_SoftmaxLayer;
};

} //namespace armnn




