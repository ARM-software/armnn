//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/Workload.hpp>
#include <backends/NeonWorkloadUtils.hpp>

#include "backends/CpuTensorHandle.hpp"
#include "backends/ArmComputeTensorUtils.hpp"
#include "backends/NeonLayerSupport.hpp"

namespace armnn
{

template<armnn::DataType dataType>
class NeonConvolution2dBaseWorkload : public TypedWorkload<Convolution2dQueueDescriptor, dataType>
{
public:
    using TypedWorkload<Convolution2dQueueDescriptor, dataType>::m_Data;

    NeonConvolution2dBaseWorkload(const Convolution2dQueueDescriptor& descriptor, const WorkloadInfo& info);

    virtual void ValidateData() const {};

protected:
    std::unique_ptr<arm_compute::IFunction> m_ConvolutionLayer;
    arm_compute::Tensor m_KernelTensor;
    arm_compute::Tensor m_BiasTensor;
};
} //namespace armnn