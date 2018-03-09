//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

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
