//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLReshapeLayer.h>

namespace armnn
{

arm_compute::Status ClReshapeWorkloadValidate(const TensorInfo& input,
                                              const TensorInfo& output);

class ClReshapeWorkload : public BaseWorkload<ReshapeQueueDescriptor>
{
public:
    ClReshapeWorkload(const ReshapeQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::CLReshapeLayer m_Layer;
};

} //namespace armnn
