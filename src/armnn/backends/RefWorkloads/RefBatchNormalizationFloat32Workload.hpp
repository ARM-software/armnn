//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"

namespace armnn
{

class RefBatchNormalizationFloat32Workload : public Float32Workload<BatchNormalizationQueueDescriptor>
{
public:
    explicit RefBatchNormalizationFloat32Workload(const BatchNormalizationQueueDescriptor& descriptor,
                                          const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    std::unique_ptr<ScopedCpuTensorHandle> m_Mean;
    std::unique_ptr<ScopedCpuTensorHandle> m_Variance;
    std::unique_ptr<ScopedCpuTensorHandle> m_Beta;
    std::unique_ptr<ScopedCpuTensorHandle> m_Gamma;
};

} //namespace armnn
