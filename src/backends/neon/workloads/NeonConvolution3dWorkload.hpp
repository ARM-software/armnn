//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
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

arm_compute::Status NeonConvolution3dWorkloadValidate(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const Convolution3dDescriptor& descriptor,
                                                      const TensorInfo& weights,
                                                      const Optional<TensorInfo>& biases,
                                                      bool isFastMathEnabled = false,
                                                      const ActivationDescriptor* activationDescriptor = nullptr);

class NeonConvolution3dWorkload : public NeonBaseWorkload<Convolution3dQueueDescriptor>
{
public:
    using BaseWorkload<Convolution3dQueueDescriptor>::m_Data;

    NeonConvolution3dWorkload(const Convolution3dQueueDescriptor& descriptor,
                              const WorkloadInfo& info,
                              std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager,
                              const bool isFastMathENabled = false);

    void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_ConvolutionLayer;
};

} //namespace armnn
