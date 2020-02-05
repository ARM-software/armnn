//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLPoolingLayer.h>

namespace armnn
{

arm_compute::Status ClPooling2dWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const Pooling2dDescriptor& descriptor);

class ClPooling2dWorkload : public BaseWorkload<Pooling2dQueueDescriptor>
{
public:
    using BaseWorkload<Pooling2dQueueDescriptor>::m_Data;

    ClPooling2dWorkload(const Pooling2dQueueDescriptor& descriptor,
                        const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::CLPoolingLayer m_PoolingLayer;
};

} //namespace armnn
