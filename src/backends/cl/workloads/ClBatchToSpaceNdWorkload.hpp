//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <arm_compute/runtime/CL/functions/CLBatchToSpaceLayer.h>

namespace armnn
{

arm_compute::Status ClBatchToSpaceNdWorkloadValidate(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const BatchToSpaceNdDescriptor& desc);

class ClBatchToSpaceNdWorkload : public BaseWorkload<BatchToSpaceNdQueueDescriptor>
{
public:
    ClBatchToSpaceNdWorkload(const BatchToSpaceNdQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:

    mutable arm_compute::CLBatchToSpaceLayer m_Layer;
};

} //namespace armnn
