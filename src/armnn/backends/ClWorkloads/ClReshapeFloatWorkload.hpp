//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

class ClReshapeFloatWorkload : public FloatWorkload<ReshapeQueueDescriptor>
{
public:
    ClReshapeFloatWorkload(const ReshapeQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::CLReshapeLayer m_Layer;
};

} //namespace armnn


