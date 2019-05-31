//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLQuantizationLayer.h>

namespace armnn
{

arm_compute::Status ClQuantizeWorkloadValidate(const TensorInfo& input,
                                               const TensorInfo& output);

class ClQuantizeWorkload : public BaseWorkload<QuantizeQueueDescriptor>
{
public:
    ClQuantizeWorkload(const QuantizeQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::CLQuantizationLayer m_Layer;
};

} //namespace armnn