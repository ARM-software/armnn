//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Tensor.hpp>

#include <vector>

namespace armnn
{

/// Contains information about inputs and outputs to a layer.
/// This is needed at construction of workloads, but are not stored.
struct WorkloadInfo
{
    std::vector<TensorInfo> m_InputTensorInfos;
    std::vector<TensorInfo> m_OutputTensorInfos;
};

} //namespace armnn
