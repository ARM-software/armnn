//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/Workload.hpp>
#include <arm_compute/runtime/CL/functions/CLPadLayer.h>

namespace armnn {

class ClPadWorkload : public BaseWorkload<PadQueueDescriptor>
{
public:
    ClPadWorkload(const PadQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::CLPadLayer m_Layer;
};

arm_compute::Status ClPadValidate(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const PadDescriptor& descriptor);

} //namespace armnn
