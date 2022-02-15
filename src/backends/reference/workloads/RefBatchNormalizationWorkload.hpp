//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

class RefBatchNormalizationWorkload : public RefBaseWorkload<BatchNormalizationQueueDescriptor>
{
public:
    explicit RefBatchNormalizationWorkload(const BatchNormalizationQueueDescriptor& descriptor,
                                           const WorkloadInfo& info);
    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)  override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
    std::unique_ptr<ScopedTensorHandle> m_Mean;
    std::unique_ptr<ScopedTensorHandle> m_Variance;
    std::unique_ptr<ScopedTensorHandle> m_Beta;
    std::unique_ptr<ScopedTensorHandle> m_Gamma;
};

} //namespace armnn
