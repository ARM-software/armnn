//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{
arm_compute::Status ClActivationWorkloadValidate(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const ActivationDescriptor& descriptor);

// Activation layer execution.
class ClActivationFloat32Workload : public FloatWorkload<ActivationQueueDescriptor>
{
public:
    ClActivationFloat32Workload(const ActivationQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::CLActivationLayer m_ActivationLayer;
};

} //namespace armnn