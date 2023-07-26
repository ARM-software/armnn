//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once
#include "ClBaseWorkload.hpp"
#include <arm_compute/runtime/CL/functions/CLTile.h>

namespace armnn
{
arm_compute::Status ClTileWorkloadValidate(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const TileDescriptor& descriptor);

class ClTileWorkload : public BaseWorkload<TileQueueDescriptor> {
public:
    ClTileWorkload(const TileQueueDescriptor &descriptor,
                   const WorkloadInfo &info,
                   const arm_compute::CLCompileContext& clCompileContext);
    void Execute() const override;

private:
    mutable arm_compute::CLTile m_Layer;
};

} //namespace armnn