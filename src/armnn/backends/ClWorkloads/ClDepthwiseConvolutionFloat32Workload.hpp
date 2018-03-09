//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

class ClDepthwiseConvolutionFloat32Workload : public Float32Workload<DepthwiseConvolution2dQueueDescriptor>
{
public:
    ClDepthwiseConvolutionFloat32Workload(const DepthwiseConvolution2dQueueDescriptor& descriptor,
                                          const WorkloadInfo& info);
    void Execute() const override;

private:
    typedef float KernelDataType;
    typedef float BiasDataType;

    mutable std::unique_ptr<arm_compute::IFunction> m_pDepthwiseConvolutionLayer;

    arm_compute::CLTensor m_KernelTensor;
    arm_compute::CLTensor m_BiasTensor;

    template <typename WorkloadType>
    friend void InitClDepthwiseConvolutionWorkload(WorkloadType& workload);
};

} //namespace armnn




