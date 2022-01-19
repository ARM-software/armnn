//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/Tensor.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>

#include <arm_compute/runtime/NEON/functions/NEDeconvolutionLayer.h>

#include <memory>

namespace armnn
{

arm_compute::Status NeonTransposeConvolution2dWorkloadValidate(const TensorInfo& input,
                                                               const TensorInfo& output,
                                                               const TransposeConvolution2dDescriptor& descriptor,
                                                               const TensorInfo& weights,
                                                               const Optional<TensorInfo>& biases);

class NeonTransposeConvolution2dWorkload : public NeonBaseWorkload<TransposeConvolution2dQueueDescriptor>
{
public:
    NeonTransposeConvolution2dWorkload(const TransposeConvolution2dQueueDescriptor& descriptor,
                                       const WorkloadInfo& info,
                                       std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);

    void Execute() const override;

private:
    std::unique_ptr<arm_compute::NEDeconvolutionLayer> m_Layer;

    std::unique_ptr<arm_compute::Tensor> m_KernelTensor;
    std::unique_ptr<arm_compute::Tensor> m_BiasTensor;

    void FreeUnusedTensors();
};

} //namespace armnn
