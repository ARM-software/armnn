//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <neon/workloads/NeonWorkloadUtils.hpp>

#include <arm_compute/runtime/NEON/functions/NEScale.h>

namespace armnn
{

arm_compute::Status NeonResizeBilinearWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class NeonResizeBilinearWorkload : public BaseWorkload<ResizeBilinearQueueDescriptor>
{
public:
    NeonResizeBilinearWorkload(const ResizeBilinearQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::NEScale m_ResizeBilinearLayer;
};

} //namespace armnn
