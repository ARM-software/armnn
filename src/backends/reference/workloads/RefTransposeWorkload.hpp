//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"

#include <armnn/TypesUtils.hpp>

namespace armnn
{

template <armnn::DataType DataType>
class RefTransposeWorkload : public TypedWorkload<TransposeQueueDescriptor, DataType>
{
public:
    static const std::string& GetName()
    {
        static const std::string name = std::string("RefTranspose") + GetDataTypeName(DataType) + "Workload";
        return name;
    }

    using TypedWorkload<TransposeQueueDescriptor, DataType>::m_Data;
    using TypedWorkload<TransposeQueueDescriptor, DataType>::TypedWorkload;
    void Execute() const override;
    void ExecuteAsync(ExecutionData& executionData)  override;
private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
};

using RefTransposeBFloat16Workload = RefTransposeWorkload<DataType::BFloat16>;
using RefTransposeFloat16Workload = RefTransposeWorkload<DataType::Float16>;
using RefTransposeFloat32Workload = RefTransposeWorkload<DataType::Float32>;
using RefTransposeQAsymmS8Workload = RefTransposeWorkload<DataType::QAsymmS8>;
using RefTransposeQAsymm8Workload = RefTransposeWorkload<DataType::QAsymmU8>;
using RefTransposeQSymm16Workload = RefTransposeWorkload<DataType::QSymmS16>;

} //namespace armnn