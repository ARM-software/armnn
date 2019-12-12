//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <neon/workloads/NeonWorkloadUtils.hpp>

#include <armnn/TypesUtils.hpp>
#include <arm_compute/runtime/IFunction.h>

#include <memory>

namespace armnn
{

arm_compute::Status NeonReshapeWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class NeonReshapeWorkload : public BaseWorkload<ReshapeQueueDescriptor>
{
public:
    NeonReshapeWorkload(const ReshapeQueueDescriptor& descriptor, const WorkloadInfo& info);

    virtual void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_Layer;
};

} //namespace armnn
