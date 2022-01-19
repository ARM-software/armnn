//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/NEON/functions/NECast.h>

namespace armnn
{

arm_compute::Status NeonCastValidate(const TensorInfo& input, const TensorInfo& output);

class NeonCastWorkload : public NeonBaseWorkload<CastQueueDescriptor>
{
public:
    NeonCastWorkload(const CastQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NECast m_CastLayer;
};

} // namespace armnn
