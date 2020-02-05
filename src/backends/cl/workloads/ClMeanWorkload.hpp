//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLReduceMean.h>

namespace armnn
{

arm_compute::Status ClMeanValidate(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const MeanDescriptor& desc);

class ClMeanWorkload : public BaseWorkload<MeanQueueDescriptor>
{
public:
    ClMeanWorkload(const MeanQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    // Not using CLMeanStdDev, as 4D input tensor support for Mean has been added to a new function called CLReduceMean.
    mutable arm_compute::CLReduceMean m_Layer;
};

} //namespace armnn
