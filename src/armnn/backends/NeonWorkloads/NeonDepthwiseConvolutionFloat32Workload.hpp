//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

namespace armnn
{

class NeonDepthwiseConvolutionFloat32Workload : public Float32Workload<DepthwiseConvolution2dQueueDescriptor>
{
public:
    NeonDepthwiseConvolutionFloat32Workload(const DepthwiseConvolution2dQueueDescriptor& descriptor,
                                            const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::IFunction> m_pDepthwiseConvolutionLayer;

    arm_compute::Tensor m_KernelTensor;
    arm_compute::Tensor m_BiasTensor;
};

} //namespace armnn




