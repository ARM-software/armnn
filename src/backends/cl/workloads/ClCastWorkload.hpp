//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/CL/functions/CLCast.h>

namespace armnn
{

arm_compute::Status ClCastValidate(const TensorInfo& input, const TensorInfo& output);

class ClCastWorkload : public ClBaseWorkload<CastQueueDescriptor>
{
public:
    ClCastWorkload(const CastQueueDescriptor& descriptor,
                   const WorkloadInfo& info,
                   const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLCast m_CastLayer;
};

} // namespace armnn
