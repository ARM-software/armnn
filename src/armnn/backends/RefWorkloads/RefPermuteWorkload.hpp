//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/Workload.hpp"

#include <armnn/TypesUtils.hpp>

namespace armnn
{

template <armnn::DataType DataType>
class RefPermuteWorkload : public TypedWorkload<PermuteQueueDescriptor, DataType>
{
public:
    static const std::string& GetName()
    {
        static const std::string name = std::string("RefPermute") + GetDataTypeName(DataType) + "Workload";
        return name;
    }

    using TypedWorkload<PermuteQueueDescriptor, DataType>::m_Data;
    using TypedWorkload<PermuteQueueDescriptor, DataType>::TypedWorkload;
    void Execute() const override;
};

using RefPermuteFloat32Workload = RefPermuteWorkload<DataType::Float32>;
using RefPermuteUint8Workload   = RefPermuteWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn
