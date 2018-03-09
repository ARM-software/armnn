//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

class ClL2NormalizationFloat32Workload : public Float32Workload<L2NormalizationQueueDescriptor>
{
public:
    ClL2NormalizationFloat32Workload(const L2NormalizationQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    // Purposely not a CLL2Normalize function. See constructor.
    mutable arm_compute::CLNormalizationLayer m_Layer;
};

} //namespace armnn




