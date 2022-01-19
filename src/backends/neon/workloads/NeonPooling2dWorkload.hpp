//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/IFunction.h>

#include <memory>

namespace armnn
{

arm_compute::Status NeonPooling2dWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const Pooling2dDescriptor& descriptor);

class NeonPooling2dWorkload : public NeonBaseWorkload<Pooling2dQueueDescriptor>
{
public:
    using BaseWorkload<Pooling2dQueueDescriptor>::m_Data;

    NeonPooling2dWorkload(const Pooling2dQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_PoolingLayer;
};

} //namespace armnn
