//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLConv3D.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>

#include <memory>

namespace armnn
{

arm_compute::Status ClConvolution3dWorkloadValidate(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    const Convolution3dDescriptor& descriptor,
                                                    const TensorInfo& weights,
                                                    const Optional<TensorInfo>& biases,
                                                    bool isFastMathEnabled = false,
                                                    const ActivationDescriptor* activationDescriptor = nullptr);

class ClConvolution3dWorkload : public ClBaseWorkload<Convolution3dQueueDescriptor>
{
public:
    ClConvolution3dWorkload(const Convolution3dQueueDescriptor& descriptor,
                            const WorkloadInfo& info,
                            std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager,
                            const arm_compute::CLCompileContext& clCompileContext,
                            const bool isFastMathEnabled = false);
    void Execute() const override;

private:
    mutable arm_compute::CLConv3D m_ConvolutionLayer;
};

} //namespace armnn

