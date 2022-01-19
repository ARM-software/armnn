//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/IFunction.h>

namespace armnn
{

arm_compute::Status NeonArgMinMaxWorkloadValidate(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const ArgMinMaxDescriptor& descriptor);

class NeonArgMinMaxWorkload : public NeonBaseWorkload<ArgMinMaxQueueDescriptor>
{
public:
    NeonArgMinMaxWorkload(const ArgMinMaxQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_ArgMinMaxLayer;
};

} //namespace armnn
