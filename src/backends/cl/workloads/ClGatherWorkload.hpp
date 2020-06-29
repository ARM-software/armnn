//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLGather.h>

namespace armnn
{
arm_compute::Status ClGatherWorkloadValidate(const TensorInfo& input,
                                             const TensorInfo& indices,
                                             const TensorInfo& output,
                                             const GatherDescriptor& descriptor);

class ClGatherWorkload : public BaseWorkload<GatherQueueDescriptor>
{
public:
    ClGatherWorkload(const GatherQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::CLGather m_Layer;
};

} // namespace armnn
