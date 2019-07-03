//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/IFunction.h>

namespace armnn
{

arm_compute::Status NeonPreluWorkloadValidate(const TensorInfo& input,
                                              const TensorInfo& alpha,
                                              const TensorInfo& output);

class NeonPreluWorkload : public BaseWorkload<PreluQueueDescriptor>
{
public:
    NeonPreluWorkload(const PreluQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_PreluLayer;
};

} //namespace armnn
