//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/runtime/NEON/functions/NEStackLayer.h>

namespace armnn
{
arm_compute::Status NeonStackWorkloadValidate(const std::vector<const TensorInfo*>& inputs,
                                              const TensorInfo& output,
                                              const StackDescriptor& descriptor);

class NeonStackWorkload : public NeonBaseWorkload<StackQueueDescriptor>
{
public:
    NeonStackWorkload(const StackQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::NEStackLayer> m_Layer;
};

} //namespace armnn
