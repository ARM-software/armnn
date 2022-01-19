//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/Tensor.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>

#include <memory>

namespace armnn
{

arm_compute::Status NeonConvolution2dWorkloadValidate(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const Convolution2dDescriptor& descriptor,
                                                      const TensorInfo& weights,
                                                      const Optional<TensorInfo>& biases,
                                                      bool isFastMathEnabled = false,
                                                      const ActivationDescriptor* activationDescriptor = nullptr);

class NeonConvolution2dWorkload : public NeonBaseWorkload<Convolution2dQueueDescriptor>
{
public:
    using BaseWorkload<Convolution2dQueueDescriptor>::m_Data;

    NeonConvolution2dWorkload(const Convolution2dQueueDescriptor& descriptor,
                              const WorkloadInfo& info,
                              std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager,
                              const bool isFastMathENabled = false);

    void Execute() const override;

    arm_compute::ConvolutionMethod GetConvolutionMethod() const;

private:
    std::unique_ptr<arm_compute::IFunction> m_ConvolutionLayer;

    std::unique_ptr<arm_compute::Tensor> m_KernelTensor;
    std::unique_ptr<arm_compute::Tensor> m_BiasTensor;

    arm_compute::ConvolutionMethod m_ConvolutionMethod;

    void FreeUnusedTensors();

};

} //namespace armnn
