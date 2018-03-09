//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"

#include <armnn/TypesUtils.hpp>
#include <arm_compute/runtime/CL/functions/CLPermute.h>

#include <string>

namespace armnn
{

arm_compute::Status ClPermuteWorkloadValidate(const PermuteDescriptor& descriptor);

template <armnn::DataType DataType>
class ClPermuteWorkload : public TypedWorkload<PermuteQueueDescriptor, DataType>
{
public:
    static const std::string& GetName()
    {
        static const std::string name = std::string("ClPermute") + GetDataTypeName(DataType) + "Workload";
        return name;
    }

    ClPermuteWorkload(const PermuteQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    using TypedWorkload<PermuteQueueDescriptor, DataType>::m_Data;
    mutable arm_compute::CLPermute m_PermuteFunction;
};

using ClPermuteFloat32Workload = ClPermuteWorkload<DataType::Float32>;
using ClPermuteUint8Workload = ClPermuteWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn
