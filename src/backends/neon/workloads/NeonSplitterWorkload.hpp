//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/IFunction.h>

#include <functional>

namespace armnn
{

arm_compute::Status NeonSplitterWorkloadValidate(const TensorInfo& input,
                                                 const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                                                 unsigned int splitAxis);

class NeonSplitterWorkload : public NeonBaseWorkload<SplitterQueueDescriptor>
{
public:
    NeonSplitterWorkload(const SplitterQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::IFunction> m_Layer;
};

} //namespace armnn
