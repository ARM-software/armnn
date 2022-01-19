//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/CL/functions/CLSplit.h>

#include <functional>

namespace armnn
{

arm_compute::Status ClSplitterWorkloadValidate(const TensorInfo& input,
                                               const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                                               unsigned int splitAxis);

class ClSplitterWorkload : public ClBaseWorkload<SplitterQueueDescriptor>
{
public:
    ClSplitterWorkload(const SplitterQueueDescriptor& descriptor,
                       const WorkloadInfo& info,
                       const arm_compute::CLCompileContext& clCompileContext);

    void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_Layer;
};

} //namespace armnn
