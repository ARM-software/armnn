//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

namespace armnn
{
class NeonL2NormalizationFloat32Workload : public Float32Workload<L2NormalizationQueueDescriptor>
{
public:
    NeonL2NormalizationFloat32Workload(const L2NormalizationQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    // Purposely not a NEL2Normalize function. See constructor.
    mutable arm_compute::NENormalizationLayer m_Layer;
};
} //namespace armnn




