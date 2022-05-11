//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/Tensor.h>

#include <memory>

namespace armnn
{

arm_compute::Status NeonDepthwiseConvolutionWorkloadValidate(const TensorInfo& input,
                                                             const TensorInfo& output,
                                                             const DepthwiseConvolution2dDescriptor& descriptor,
                                                             const TensorInfo& weights,
                                                             const Optional<TensorInfo>& biases,
                                                             const ActivationDescriptor* activationDescriptor
                                                                     = nullptr);

class NeonDepthwiseConvolutionWorkload : public NeonBaseWorkload<DepthwiseConvolution2dQueueDescriptor>
{
public:
    NeonDepthwiseConvolutionWorkload(const DepthwiseConvolution2dQueueDescriptor& descriptor,
                                     const WorkloadInfo& info);

    virtual void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::IFunction> m_pDepthwiseConvolutionLayer;
};

} // namespace armnn
