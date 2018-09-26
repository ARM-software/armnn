//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/Workload.hpp>

#include <arm_compute/runtime/CL/CLFunctions.h>

namespace armnn
{

// Activation layer execution.
class ClActivationUint8Workload : public Uint8Workload<ActivationQueueDescriptor>
{
public:
    ClActivationUint8Workload(const ActivationQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::CLActivationLayer m_ActivationLayer;
};

} //namespace armnn



