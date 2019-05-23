//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/CLFunctions.h>

#include <functional>

namespace armnn
{

arm_compute::Status ClSplitterWorkloadValidate(const TensorInfo& input,
                                               const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                                               unsigned int splitAxis);

class ClSplitterWorkload : public BaseWorkload<SplitterQueueDescriptor>
{
public:
    ClSplitterWorkload(const SplitterQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::CLSplit> m_Layer;
};

} //namespace armnn
