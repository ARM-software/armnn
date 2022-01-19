//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/NEON/functions/NEChannelShuffleLayer.h>

namespace armnn
{

arm_compute::Status NeonChannelShuffleValidate(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const ChannelShuffleDescriptor& descriptor);

class NeonChannelShuffleWorkload : public NeonBaseWorkload<ChannelShuffleQueueDescriptor>
{
public:
    NeonChannelShuffleWorkload(const ChannelShuffleQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEChannelShuffleLayer m_ChannelShuffleLayer;
};

} // namespace armnn
