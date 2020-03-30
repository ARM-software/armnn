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

class NeonConvertBf16ToFp32Workload : public BFloat16ToFloat32Workload<ConvertBf16ToFp32QueueDescriptor>
{
public:
    NeonConvertBf16ToFp32Workload(const ConvertBf16ToFp32QueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    using TensorHandlePair = std::pair<const ITensorHandle*, ITensorHandle*>;
    std::vector<TensorHandlePair> m_TensorHandlePairs;
};

} //namespace armnn
