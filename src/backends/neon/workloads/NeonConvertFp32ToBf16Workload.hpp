//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <neon/workloads/NeonWorkloadUtils.hpp>

namespace armnn
{

class NeonConvertFp32ToBf16Workload : public Float32ToBFloat16Workload<ConvertFp32ToBf16QueueDescriptor>
{
public:
    NeonConvertFp32ToBf16Workload(const ConvertFp32ToBf16QueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    using TensorHandlePair = std::pair<const ITensorHandle*, ITensorHandle*>;
    std::vector<TensorHandlePair> m_TensorHandlePairs;
};

} //namespace armnn
