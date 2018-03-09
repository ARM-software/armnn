//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

namespace armnn
{

class NeonMultiplicationFloat32Workload : public Float32Workload<MultiplicationQueueDescriptor>
{
public:
    NeonMultiplicationFloat32Workload(const MultiplicationQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEPixelWiseMultiplication m_PixelWiseMultiplication;
};

} //namespace armnn




