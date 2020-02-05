//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLNormalizationLayer.h>

namespace armnn
{

arm_compute::Status ClNormalizationWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const NormalizationDescriptor& descriptor);

class ClNormalizationFloatWorkload : public FloatWorkload<NormalizationQueueDescriptor>
{
public:
    ClNormalizationFloatWorkload(const NormalizationQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::CLNormalizationLayer    m_NormalizationLayer;
};

} //namespace armnn
