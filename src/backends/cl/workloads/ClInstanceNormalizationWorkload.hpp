//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLInstanceNormalizationLayer.h>

namespace armnn
{

arm_compute::Status ClInstanceNormalizationWorkloadValidate(const TensorInfo& input,
                                                            const TensorInfo& output,
                                                            const InstanceNormalizationDescriptor& descriptor);

class ClInstanceNormalizationWorkload : public ClBaseWorkload<InstanceNormalizationQueueDescriptor>
{
public:
    ClInstanceNormalizationWorkload(const InstanceNormalizationQueueDescriptor& descriptor,
                                    const WorkloadInfo& info,
                                    const arm_compute::CLCompileContext& clCompileContext);
    void Execute() const override;

private:
    mutable arm_compute::CLInstanceNormalizationLayer m_Layer;
};

} // namespace armnn
