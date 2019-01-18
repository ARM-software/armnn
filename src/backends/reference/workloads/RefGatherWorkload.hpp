//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

#include <armnn/TypesUtils.hpp>

namespace armnn
{

template <armnn::DataType DataType>
class RefGatherWorkload : public FirstInputTypedWorkload<GatherQueueDescriptor, DataType>
{
public:

    static const std::string& GetName()
    {
        static const std::string name = std::string("RefGather") + GetDataTypeName(DataType) + "Workload";
        return name;
    }

    using FirstInputTypedWorkload<GatherQueueDescriptor, DataType>::m_Data;
    using FirstInputTypedWorkload<GatherQueueDescriptor, DataType>::FirstInputTypedWorkload;

    void Execute() const override;
};

using RefGatherFloat32Workload = RefGatherWorkload<DataType::Float32>;
using RefGatherUint8Workload = RefGatherWorkload<DataType::QuantisedAsymm8>;

} // namespace armnn
