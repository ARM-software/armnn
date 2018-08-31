//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

namespace armnn
{

class NeonReshapeFloatWorkload : public FloatWorkload<ReshapeQueueDescriptor>
{
public:
    NeonReshapeFloatWorkload(const ReshapeQueueDescriptor& descriptor, const WorkloadInfo& info);

    virtual void Execute() const override;

private:
    mutable arm_compute::NEReshapeLayer m_Layer;
};

} //namespace armnn





