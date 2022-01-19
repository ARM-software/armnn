//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLStackLayer.h>

namespace armnn
{
arm_compute::Status ClStackWorkloadValidate(const std::vector<const TensorInfo*>& inputs,
                                            const TensorInfo& output,
                                            const StackDescriptor& descriptor);

class ClStackWorkload : public ClBaseWorkload<StackQueueDescriptor>
{
public:
    ClStackWorkload(const StackQueueDescriptor& descriptor,
                    const WorkloadInfo& info,
                    const arm_compute::CLCompileContext& clCompileContext);

    void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::CLStackLayer> m_Layer;
};

} //namespace armnn
