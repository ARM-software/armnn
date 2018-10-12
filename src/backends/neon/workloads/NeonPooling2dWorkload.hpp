//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/neon/workloads/NeonWorkloadUtils.hpp>

namespace armnn
{

arm_compute::Status NeonPooling2dWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const Pooling2dDescriptor& descriptor);

class NeonPooling2dWorkload : public BaseWorkload<Pooling2dQueueDescriptor>
{
public:
    using BaseWorkload<Pooling2dQueueDescriptor>::m_Data;

    NeonPooling2dWorkload(const Pooling2dQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::NEPoolingLayer m_PoolingLayer;
};

} //namespace armnn
