//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/CLFunctions.h>

namespace armnn
{

arm_compute::Status ClMergerWorkloadValidate(const std::vector<const TensorInfo*>& inputs,
                                             const TensorInfo& output,
                                             const MergerDescriptor& descriptor);

class ClMergerWorkload : public BaseWorkload<MergerQueueDescriptor>
{
public:
    ClMergerWorkload(const MergerQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::CLConcatenateLayer> m_Layer;
};

} //namespace armnn
