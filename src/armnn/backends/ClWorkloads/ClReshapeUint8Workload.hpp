//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

// Reshape
class ClReshapeUint8Workload : public Uint8Workload<ReshapeQueueDescriptor>
{
public:
    ClReshapeUint8Workload( const ReshapeQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::CLReshapeLayer m_Layer;
};

} //namespace armnn


