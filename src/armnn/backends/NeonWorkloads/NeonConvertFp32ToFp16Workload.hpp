//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"
#include "backends/NeonWorkloadUtils.hpp"

namespace armnn
{

class NeonConvertFp32ToFp16Workload : public Float32ToFloat16Workload<ConvertFp32ToFp16QueueDescriptor>
{
public:
    NeonConvertFp32ToFp16Workload(const ConvertFp32ToFp16QueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    using TensorHandlePair = std::pair<const ITensorHandle*, ITensorHandle*>;
    std::vector<TensorHandlePair> m_TensorHandlePairs;
};

} //namespace armnn
