//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/IFunction.h>

#include <memory>

namespace armnn
{

arm_compute::Status NeonSubtractionWorkloadValidate(const TensorInfo& input0,
                                                    const TensorInfo& input1,
                                                    const TensorInfo& output);

class NeonSubtractionWorkload : public BaseWorkload<SubtractionQueueDescriptor>
{
public:
    NeonSubtractionWorkload(const SubtractionQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_SubLayer;
};

} //namespace armnn
