//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/NEON/functions/NEConcatenateLayer.h>

#include <memory>

namespace armnn
{
arm_compute::Status NeonMergerWorkloadValidate(const std::vector<const TensorInfo*>& inputs,
                                               const TensorInfo& output,
                                               const MergerDescriptor& descriptor);

class NeonMergerWorkload : public BaseWorkload<MergerQueueDescriptor>
{
public:
    NeonMergerWorkload(const MergerQueueDescriptor& descriptor, const WorkloadInfo& info);

    using BaseWorkload<MergerQueueDescriptor>::BaseWorkload;
    void Execute() const override;

private:
    std::unique_ptr<arm_compute::NEConcatenateLayer> m_Layer;
};

} //namespace armnn
