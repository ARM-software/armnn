//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLReduceMean.h>

namespace armnn
{

arm_compute::Status ClMeanValidate(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const MeanDescriptor& descriptor);

class ClMeanWorkload : public ClBaseWorkload<MeanQueueDescriptor>
{
public:
    ClMeanWorkload(const MeanQueueDescriptor& descriptor,
                   const WorkloadInfo& info,
                   const arm_compute::CLCompileContext& clCompileContext);

    void Execute() const override;

private:
    // Not using CLMeanStdDev, as 4D input tensor support for Mean has been added to a new function called CLReduceMean.
    mutable arm_compute::CLReduceMean m_Layer;
};

} //namespace armnn
