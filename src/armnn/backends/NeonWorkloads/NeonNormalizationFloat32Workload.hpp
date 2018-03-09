//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

namespace armnn
{

class NeonNormalizationFloat32Workload : public Float32Workload<NormalizationQueueDescriptor>
{
public:
    NeonNormalizationFloat32Workload(const NormalizationQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NENormalizationLayer m_NormalizationLayer;
};

} //namespace armnn




