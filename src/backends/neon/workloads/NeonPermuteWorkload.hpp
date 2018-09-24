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

template <armnn::DataType... DataTypes>
class NeonPermuteWorkload : public TypedWorkload<PermuteQueueDescriptor, DataTypes...>
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
    using TypedWorkload<PermuteQueueDescriptor, DataTypes...>::m_Data;
    mutable arm_compute::NEPermute m_PermuteFunction;
};

using NeonPermuteFloatWorkload = NeonPermuteWorkload<DataType::Float16, DataType::Float32>;
using NeonPermuteUint8Workload = NeonPermuteWorkload<DataType::QuantisedAsymm8>;

} // namespace armnn
