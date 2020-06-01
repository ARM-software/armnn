//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefBatchNormalizationWorkload : public BaseWorkload<BatchNormalizationQueueDescriptor>
{
public:
    explicit RefBatchNormalizationWorkload(const BatchNormalizationQueueDescriptor& descriptor,
                                           const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    std::unique_ptr<ScopedCpuTensorHandle> m_Mean;
    std::unique_ptr<ScopedCpuTensorHandle> m_Variance;
    std::unique_ptr<ScopedCpuTensorHandle> m_Beta;
    std::unique_ptr<ScopedCpuTensorHandle> m_Gamma;
};

} //namespace armnn
