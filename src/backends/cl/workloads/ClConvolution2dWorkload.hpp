//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>
#include <armnn/Descriptors.hpp>

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLConvolutionLayer.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>

#include <memory>

namespace armnn
{

arm_compute::Status ClConvolution2dWorkloadValidate(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    const Convolution2dDescriptor& descriptor,
                                                    const TensorInfo& weights,
                                                    const Optional<TensorInfo>& biases,
                                                    bool isFastMathEnabled = false,
                                                    const ActivationDescriptor* activationDescriptor = nullptr);

class ClConvolution2dWorkload : public BaseWorkload<Convolution2dQueueDescriptor>
{
public:
    ClConvolution2dWorkload(const Convolution2dQueueDescriptor& descriptor,
                            const WorkloadInfo& info,
                            std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager,
                            const bool isFastMathEnabled = false);
    void Execute() const override;

    arm_compute::ConvolutionMethod GetConvolutionMethod() const;

private:
    mutable arm_compute::CLConvolutionLayer m_ConvolutionLayer;

    std::unique_ptr<arm_compute::CLTensor> m_KernelTensor;
    std::unique_ptr<arm_compute::CLTensor> m_BiasTensor;

    arm_compute::ConvolutionMethod m_ConvolutionMethod;

    void FreeUnusedTensors();
};

} //namespace armnn

