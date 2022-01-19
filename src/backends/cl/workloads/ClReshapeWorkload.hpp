//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLReshapeLayer.h>

namespace armnn
{

arm_compute::Status ClReshapeWorkloadValidate(const TensorInfo& input,
                                              const TensorInfo& output);

class ClReshapeWorkload : public ClBaseWorkload<ReshapeQueueDescriptor>
{
public:
    ClReshapeWorkload(const ReshapeQueueDescriptor& descriptor,
                      const WorkloadInfo& info,
                      const arm_compute::CLCompileContext& clCompileContext);

    void Execute() const override;

private:
    mutable arm_compute::CLReshapeLayer m_Layer;
};

} //namespace armnn
