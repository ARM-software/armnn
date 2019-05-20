//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/NEON/NEFunctions.h>

#include <functional>

namespace armnn
{

arm_compute::Status NeonSplitterWorkloadValidate(const TensorInfo& input,
                                                 const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                                                 unsigned int splitAxis);

class NeonSplitterWorkload : public BaseWorkload<SplitterQueueDescriptor>
{
public:
    NeonSplitterWorkload(const SplitterQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::NESplit> m_Layer;
};

} //namespace armnn
