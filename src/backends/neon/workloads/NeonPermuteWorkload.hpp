//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/Workload.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/neon/workloads/NeonWorkloadUtils.hpp>

#include <armnn/TypesUtils.hpp>
#include <arm_compute/runtime/NEON/functions/NEPermute.h>

#include <string>

namespace armnn
{
arm_compute::Status NeonPermuteWorkloadValidate(const TensorInfo& input, const TensorInfo& output,
                                                const PermuteDescriptor& descriptor);

class NeonPermuteWorkload : public BaseWorkload<PermuteQueueDescriptor>
{
public:
    static const std::string& GetName()
    {
        static const std::string name = std::string("NeonPermuteWorkload");
        return name;
    }

    NeonPermuteWorkload(const PermuteQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    using BaseWorkload<PermuteQueueDescriptor>::m_Data;
    mutable arm_compute::NEPermute m_PermuteFunction;
};

} // namespace armnn
