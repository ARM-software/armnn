//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/WorkloadData.hpp>
#include "ClBaseWorkload.hpp"
#include <arm_compute/runtime/CL/functions/CLPadLayer.h>

namespace armnn {

class ClPadWorkload : public ClBaseWorkload<PadQueueDescriptor>
{
public:
    ClPadWorkload(const PadQueueDescriptor& descriptor,
                  const WorkloadInfo& info,
                  const arm_compute::CLCompileContext& clCompileContext);

    void Execute() const override;

private:
    mutable arm_compute::CLPadLayer m_Layer;
};

arm_compute::Status ClPadValidate(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const PadDescriptor& descriptor);

} //namespace armnn
