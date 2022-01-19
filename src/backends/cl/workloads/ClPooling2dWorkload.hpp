//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLPoolingLayer.h>

namespace armnn
{

arm_compute::Status ClPooling2dWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const Pooling2dDescriptor& descriptor);

class ClPooling2dWorkload : public ClBaseWorkload<Pooling2dQueueDescriptor>
{
public:
    using BaseWorkload<Pooling2dQueueDescriptor>::m_Data;

    ClPooling2dWorkload(const Pooling2dQueueDescriptor& descriptor,
                        const WorkloadInfo& info,
                        const arm_compute::CLCompileContext& clCompileContext);

    void Execute() const override;

private:
    mutable arm_compute::CLPoolingLayer m_PoolingLayer;
};

} //namespace armnn
