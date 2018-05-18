//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

#include "arm_compute/runtime/MemoryManagerOnDemand.h"

#include <memory>

namespace armnn
{

class ClConvolution2dUint8Workload : public Uint8Workload<Convolution2dQueueDescriptor>
{
public:
    ClConvolution2dUint8Workload(const Convolution2dQueueDescriptor& descriptor, const WorkloadInfo& info,
                                 std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);
    void Execute() const override;

private:
    mutable arm_compute::CLConvolutionLayer         m_ConvolutionLayer;

    arm_compute::CLTensor m_KernelTensor;
    arm_compute::CLTensor m_BiasTensor;
};

} //namespace armnn

