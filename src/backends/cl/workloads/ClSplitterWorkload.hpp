//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/IFunction.h>

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
    std::unique_ptr<arm_compute::IFunction> m_Layer;
};

} //namespace armnn
