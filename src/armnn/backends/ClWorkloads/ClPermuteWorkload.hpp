//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"
#include "backends/ClWorkloadUtils.hpp"

#include <armnn/TypesUtils.hpp>
#include <arm_compute/runtime/CL/functions/CLPermute.h>

#include <string>

namespace armnn
{

arm_compute::Status ClPermuteWorkloadValidate(const PermuteDescriptor& descriptor);

template<armnn::DataType... DataTypes>
class ClPermuteWorkload : public TypedWorkload<PermuteQueueDescriptor, DataTypes...>
{
public:
    static const std::string& GetName()
    {
        static const std::string name = std::string("ClPermuteWorkload");
        return name;
    }

    ClPermuteWorkload(const PermuteQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    using TypedWorkload<PermuteQueueDescriptor, DataTypes...>::m_Data;
    mutable arm_compute::CLPermute m_PermuteFunction;
};

using ClPermuteFloatWorkload = ClPermuteWorkload<DataType::Float16, DataType::Float32>;
using ClPermuteUint8Workload = ClPermuteWorkload<DataType::QuantisedAsymm8>;

} // namespace armnn
