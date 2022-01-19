//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

#include <armnn/TypesUtils.hpp>
#include <arm_compute/runtime/CL/functions/CLPermute.h>

#include <string>

namespace armnn
{

arm_compute::Status ClPermuteWorkloadValidate(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const PermuteDescriptor& descriptor);

class ClPermuteWorkload : public ClBaseWorkload<PermuteQueueDescriptor>
{
public:
    static const std::string& GetName()
    {
        static const std::string name = std::string("ClPermuteWorkload");
        return name;
    }

    ClPermuteWorkload(const PermuteQueueDescriptor& descriptor,
                      const WorkloadInfo& info,
                      const arm_compute::CLCompileContext& clCompileContext);
    void Execute() const override;

private:
    using BaseWorkload<PermuteQueueDescriptor>::m_Data;
    mutable arm_compute::CLPermute m_PermuteFunction;
};

} // namespace armnn
