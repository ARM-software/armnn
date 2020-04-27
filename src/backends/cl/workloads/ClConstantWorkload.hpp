//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <arm_compute/core/Error.h>
#include <backendsCommon/Workload.hpp>

namespace armnn
{
arm_compute::Status ClConstantWorkloadValidate(const TensorInfo& output);

class ClConstantWorkload : public BaseWorkload<ConstantQueueDescriptor>
{
public:
    ClConstantWorkload(const ConstantQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable bool m_RanOnce;
};

} //namespace armnn
