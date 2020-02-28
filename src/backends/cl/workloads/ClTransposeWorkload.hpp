//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

#include <armnn/TypesUtils.hpp>
#include <arm_compute/runtime/CL/functions/CLPermute.h>

#include <string>

namespace armnn
{

arm_compute::Status ClTransposeWorkloadValidate(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const TransposeDescriptor& descriptor);

class ClTransposeWorkload : public BaseWorkload<TransposeQueueDescriptor>
{
public:
    static const std::string& GetName()
    {
        static const std::string name = std::string("ClTransposeWorkload");
        return name;
    }

    ClTransposeWorkload(const TransposeQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    using BaseWorkload<TransposeQueueDescriptor>::m_Data;
    mutable arm_compute::CLPermute m_PermuteFunction;
};

} // namespace armnn
