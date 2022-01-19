//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLQuantizationLayer.h>

namespace armnn
{

arm_compute::Status ClQuantizeWorkloadValidate(const TensorInfo& input,
                                               const TensorInfo& output);

class ClQuantizeWorkload : public ClBaseWorkload<QuantizeQueueDescriptor>
{
public:
    ClQuantizeWorkload(const QuantizeQueueDescriptor& descriptor,
                       const WorkloadInfo& info,
                       const arm_compute::CLCompileContext& clCompileContext);
    void Execute() const override;

private:
    mutable arm_compute::CLQuantizationLayer m_Layer;
};

} //namespace armnn