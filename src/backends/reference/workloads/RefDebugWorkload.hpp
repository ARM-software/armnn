//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/TypesUtils.hpp>

#include <backendsCommon/Workload.hpp>

namespace armnn
{

template <armnn::DataType DataType>
class RefDebugWorkload : public TypedWorkload<DebugQueueDescriptor, DataType>
{
public:
    static const std::string& GetName()
    {
        static const std::string name = std::string("RefDebug") + GetDataTypeName(DataType) + "Workload";
        return name;
    }

    using TypedWorkload<DebugQueueDescriptor, DataType>::m_Data;
    using TypedWorkload<DebugQueueDescriptor, DataType>::TypedWorkload;

    void Execute() const override;
};

using RefDebugFloat32Workload = RefDebugWorkload<DataType::Float32>;
using RefDebugUint8Workload = RefDebugWorkload<DataType::QuantisedAsymm8>;

} // namespace armnn
