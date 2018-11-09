//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "backendsCommon/Workload.hpp"

#include <armnn/TypesUtils.hpp>

namespace armnn
{

template <armnn::DataType DataType>
class RefSpaceToBatchNdWorkload : public TypedWorkload<SpaceToBatchNdQueueDescriptor, DataType>
{
public:
    static const std::string& GetName()
    {
        static const std::string name = std::string("RefSpaceToBatchNd") + GetDataTypeName(DataType) + "Workload";
        return name;
    }

    using TypedWorkload<SpaceToBatchNdQueueDescriptor, DataType>::m_Data;
    using TypedWorkload<SpaceToBatchNdQueueDescriptor, DataType>::TypedWorkload;

    void Execute() const override;
};

using RefSpaceToBatchNdFloat32Workload = RefSpaceToBatchNdWorkload<DataType::Float32>;
using RefSpaceToBatchNdUint8Workload = RefSpaceToBatchNdWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn
