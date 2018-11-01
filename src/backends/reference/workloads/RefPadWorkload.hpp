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
class RefPadWorkload : public TypedWorkload<PadQueueDescriptor, DataType>
{
public:

    static const std::string& GetName()
    {
        static const std::string name = std::string("RefPad") + GetDataTypeName(DataType) + "Workload";
        return name;
    }

    using TypedWorkload<PadQueueDescriptor, DataType>::m_Data;
    using TypedWorkload<PadQueueDescriptor, DataType>::TypedWorkload;

    void Execute() const override;
};

using RefPadFloat32Workload = RefPadWorkload<DataType::Float32>;
using RefPadUint8Workload   = RefPadWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn
