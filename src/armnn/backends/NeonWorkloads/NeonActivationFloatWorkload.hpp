//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

namespace armnn
{

arm_compute::Status NeonActivationWorkloadValidate(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const ActivationDescriptor& descriptor);

class NeonActivationFloatWorkload : public FloatWorkload<ActivationQueueDescriptor>
{
public:
    NeonActivationFloatWorkload(const ActivationQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::NEActivationLayer m_ActivationLayer;
};
} //namespace armnn



