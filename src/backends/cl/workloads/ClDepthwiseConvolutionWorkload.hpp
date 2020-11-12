//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/CL/CLTensor.h>

namespace armnn
{

arm_compute::Status ClDepthwiseConvolutionWorkloadValidate(const TensorInfo& input,
                                                           const TensorInfo& output,
                                                           const DepthwiseConvolution2dDescriptor& descriptor,
                                                           const TensorInfo& weights,
                                                           const Optional<TensorInfo>& biases,
                                                           const ActivationDescriptor* activationDescriptor = nullptr);

class ClDepthwiseConvolutionWorkload : public BaseWorkload<DepthwiseConvolution2dQueueDescriptor>
{
public:
    using BaseWorkload<DepthwiseConvolution2dQueueDescriptor>::m_Data;

    ClDepthwiseConvolutionWorkload(const DepthwiseConvolution2dQueueDescriptor& descriptor,
                                   const WorkloadInfo& info);

    void Execute() const override;

protected:
    std::unique_ptr<arm_compute::IFunction> m_DepthwiseConvolutionLayer;

    std::unique_ptr<arm_compute::CLTensor> m_KernelTensor;
    std::unique_ptr<arm_compute::CLTensor> m_BiasTensor;

    void FreeUnusedTensors();
};

} //namespace armnn
