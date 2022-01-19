//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"
#include <arm_compute/runtime/CL/functions/CLBatchToSpaceLayer.h>

namespace armnn
{

arm_compute::Status ClBatchToSpaceNdWorkloadValidate(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const BatchToSpaceNdDescriptor& descriptor);

class ClBatchToSpaceNdWorkload : public ClBaseWorkload<BatchToSpaceNdQueueDescriptor>
{
public:
    ClBatchToSpaceNdWorkload(const BatchToSpaceNdQueueDescriptor& descriptor,
                             const WorkloadInfo& info,
                             const arm_compute::CLCompileContext& clCompileContext);

    void Execute() const override;

private:

    mutable arm_compute::CLBatchToSpaceLayer m_Layer;
};

} //namespace armnn
