//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

namespace armnn
{
arm_compute::Status NeonMultiplicationWorkloadValidate(const TensorInfo& input0,
                                                       const TensorInfo& input1,
                                                       const TensorInfo& output);

class NeonMultiplicationFloat32Workload : public FloatWorkload<MultiplicationQueueDescriptor>
{
public:
    NeonMultiplicationFloat32Workload(const MultiplicationQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEPixelWiseMultiplication m_PixelWiseMultiplication;
};

} //namespace armnn




