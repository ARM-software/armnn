//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/WorkloadData.hpp>
#include <armnn/Tensor.hpp>

namespace armnn
{

template <typename DataType>
void CopyValue(const DataType& source, const TensorInfo& sourceInfo, DataType& dest, const TensorInfo& destInfo);

template <typename DataType>
void Merger(const MergerQueueDescriptor& data);

} //namespace armnn
