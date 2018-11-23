//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <armnn/TypesUtils.hpp>

namespace armnn
{

template <armnn::DataType DataType>
class RefStridedSliceWorkload : public TypedWorkload<StridedSliceQueueDescriptor, DataType>
{
public:
    static const std::string& GetName()
    {
        static const std::string name = std::string("RefStridedSlice") + GetDataTypeName(DataType) + "Workload";
        return name;
    }

    using TypedWorkload<StridedSliceQueueDescriptor, DataType>::m_Data;
    using TypedWorkload<StridedSliceQueueDescriptor, DataType>::TypedWorkload;

    void Execute() const override;
};

using RefStridedSliceFloat32Workload = RefStridedSliceWorkload<DataType::Float32>;
using RefStridedSliceUint8Workload = RefStridedSliceWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn
