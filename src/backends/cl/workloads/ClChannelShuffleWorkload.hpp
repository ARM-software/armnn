//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/CL/functions/CLChannelShuffleLayer.h>

namespace armnn
{

arm_compute::Status ClChannelShuffleValidate(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const ChannelShuffleDescriptor& descriptor);

class ClChannelShuffleWorkload : public ClBaseWorkload<ChannelShuffleQueueDescriptor>
{
public:
    ClChannelShuffleWorkload(const ChannelShuffleQueueDescriptor& descriptor,
                             const WorkloadInfo& info,
                             const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLChannelShuffleLayer m_ChannelShuffleLayer;
};

} // namespace armnn
