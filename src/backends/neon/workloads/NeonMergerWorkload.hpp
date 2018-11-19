//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <neon/workloads/NeonWorkloadUtils.hpp>

namespace armnn
{
arm_compute::Status NeonMergerWorkloadValidate(const std::vector<const TensorInfo*>& inputs,
                                               const TensorInfo& output,
                                               const MergerDescriptor& descriptor);

class NeonMergerWorkload : public BaseWorkload<MergerQueueDescriptor>
{
public:
    NeonMergerWorkload(const MergerQueueDescriptor& descriptor, const WorkloadInfo& info);

    using BaseWorkload<MergerQueueDescriptor>::BaseWorkload;
    void Execute() const override;

private:
    mutable arm_compute::NEConcatenateLayer m_Layer;
    bool m_Execute;

};

} //namespace armnn
