//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

arm_compute::Status ClL2NormalizationWorkloadValidate(const TensorInfo& input,
                                                      const TensorInfo& output);

class ClL2NormalizationFloatWorkload : public FloatWorkload<L2NormalizationQueueDescriptor>
{
public:
    ClL2NormalizationFloatWorkload(const L2NormalizationQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    // Purposely not a CLL2Normalize function. See constructor.
    mutable arm_compute::CLNormalizationLayer m_Layer;
};

} //namespace armnn




