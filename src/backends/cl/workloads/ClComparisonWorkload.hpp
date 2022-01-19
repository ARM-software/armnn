//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLComparison.h>

namespace armnn
{

arm_compute::Status ClComparisonWorkloadValidate(const TensorInfo& input0,
                                                 const TensorInfo& input1,
                                                 const TensorInfo& output,
                                                 const ComparisonDescriptor& descriptor);

class ClComparisonWorkload : public ClBaseWorkload<ComparisonQueueDescriptor>
{
public:
    ClComparisonWorkload(const ComparisonQueueDescriptor& descriptor,
                         const WorkloadInfo& info,
                         const arm_compute::CLCompileContext& clCompileContext);
    void Execute() const override;

private:
    mutable arm_compute::CLComparison m_ComparisonLayer;
};

} //namespace armnn
