//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/CL/functions/CLConcatenateLayer.h>

namespace armnn
{

arm_compute::Status ClConcatWorkloadValidate(const std::vector<const TensorInfo*>& inputs,
                                             const TensorInfo& output,
                                             const OriginsDescriptor& descriptor);

class ClConcatWorkload : public ClBaseWorkload<ConcatQueueDescriptor>
{
public:
    ClConcatWorkload(const ConcatQueueDescriptor& descriptor,
                     const WorkloadInfo& info,
                     const arm_compute::CLCompileContext& clCompileContext);

    void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::IFunction> m_Layer;
};

} //namespace armnn
