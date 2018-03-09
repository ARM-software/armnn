//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"

#include <armnn/TypesUtils.hpp>
#include <arm_compute/runtime/NEON/functions/NEPermute.h>

#include <string>

namespace armnn
{
arm_compute::Status NeonPermuteWorkloadValidate(const TensorInfo& input, const TensorInfo& output,
                                                const PermuteDescriptor& descriptor);

template <armnn::DataType DataType>
class NeonPermuteWorkload : public TypedWorkload<PermuteQueueDescriptor, DataType>
{
public:
    static const std::string& GetName()
    {
        static const std::string name = std::string("NeonPermute") + GetDataTypeName(DataType) + "Workload";
        return name;
    }

    NeonPermuteWorkload(const PermuteQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    using TypedWorkload<PermuteQueueDescriptor, DataType>::m_Data;
    mutable arm_compute::NEPermute m_PermuteFunction;
};

using NeonPermuteFloat32Workload = NeonPermuteWorkload<DataType::Float32>;
using NeonPermuteUint8Workload = NeonPermuteWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn
