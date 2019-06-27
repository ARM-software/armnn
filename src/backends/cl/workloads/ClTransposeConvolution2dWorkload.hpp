//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>
#include <armnn/Descriptors.hpp>

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLDeconvolutionLayer.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>

#include <memory>

namespace armnn
{

arm_compute::Status ClTransposeConvolution2dWorkloadValidate(const TensorInfo& input,
                                                             const TensorInfo& output,
                                                             const TransposeConvolution2dDescriptor& descriptor,
                                                             const TensorInfo& weights,
                                                             const Optional<TensorInfo>& biases);

class ClTransposeConvolution2dWorkload : public BaseWorkload<TransposeConvolution2dQueueDescriptor>
{
public:
    ClTransposeConvolution2dWorkload(const TransposeConvolution2dQueueDescriptor& descriptor,
                                     const WorkloadInfo& info,
                                     std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);

    void Execute() const override;

private:
    mutable arm_compute::CLDeconvolutionLayer m_Layer;

    std::unique_ptr<arm_compute::CLTensor> m_WeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_BiasesTensor;

    void FreeUnusedTensors();
};

} // namespace armnn

