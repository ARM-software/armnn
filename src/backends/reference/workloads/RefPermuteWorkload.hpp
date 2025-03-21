//
// Copyright © 2017-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"

#include <armnn/TypesUtils.hpp>

namespace armnn
{

template <armnn::DataType DataType>
class RefPermuteWorkload : public TypedWorkload<PermuteQueueDescriptor, DataType>
{
public:
    using TypedWorkload<PermuteQueueDescriptor, DataType>::m_Data;
    using TypedWorkload<PermuteQueueDescriptor, DataType>::TypedWorkload;
    void Execute() const override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
};

using RefPermuteBFloat16Workload = RefPermuteWorkload<DataType::BFloat16>;
using RefPermuteFloat16Workload  = RefPermuteWorkload<DataType::Float16>;
using RefPermuteFloat32Workload  = RefPermuteWorkload<DataType::Float32>;
using RefPermuteQAsymmS8Workload = RefPermuteWorkload<DataType::QAsymmS8>;
using RefPermuteQAsymm8Workload  = RefPermuteWorkload<DataType::QAsymmU8>;
using RefPermuteQSymm16Workload  = RefPermuteWorkload<DataType::QSymmS16>;

} //namespace armnn