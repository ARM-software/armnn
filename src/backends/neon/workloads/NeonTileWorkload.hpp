//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "NeonBaseWorkload.hpp"
#include <arm_compute/runtime/NEON/functions/NETile.h>

namespace armnn
{
arm_compute::Status NeonTileWorkloadValidate(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const TileDescriptor& descriptor);

class NeonTileWorkload : public BaseWorkload<TileQueueDescriptor>
{
public:
    NeonTileWorkload(const TileQueueDescriptor &descriptor,
                     const WorkloadInfo &info);
    void Execute() const override;

private:
    mutable arm_compute::NETile m_Layer;
};

} //namespace armnn