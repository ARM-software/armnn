//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/NEON/functions/NEStridedSlice.h>

#include <memory>


namespace armnn
{

arm_compute::Status NeonStridedSliceWorkloadValidate(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const StridedSliceDescriptor& descriptor);

class NeonStridedSliceWorkload : public NeonBaseWorkload<StridedSliceQueueDescriptor>
{
public:
    NeonStridedSliceWorkload(const StridedSliceQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::NEStridedSlice> m_Layer;
};

} //namespace armnn