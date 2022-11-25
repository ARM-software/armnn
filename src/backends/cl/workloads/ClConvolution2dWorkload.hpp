//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>
#include <armnn/Descriptors.hpp>

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLConvolutionLayer.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>

#include <cl/ICLTensorProxy.hpp>

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

class ClConvolution2dWorkload : public ClBaseWorkload<Convolution2dQueueDescriptor>
{
public:
    ClConvolution2dWorkload(const Convolution2dQueueDescriptor& descriptor,
                            const WorkloadInfo& info,
                            std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager,
                            const arm_compute::CLCompileContext& clCompileContext,
                            const bool isFastMathEnabled = false);
    void Execute() const override;

    arm_compute::ConvolutionMethod GetConvolutionMethod() const;

    bool SupportsTensorHandleReplacement() const override
    {
        // NCHW DataLayout on ACL still uses paddding for alignment on the Conv2d workload so importing is unreliable.
        if (m_Data.m_Parameters.m_DataLayout == DataLayout::NCHW)
        {
            return false;
        }
        else
        {
            return true;
        }
    }


protected:
    void Reconfigure() override;

private:
    mutable arm_compute::CLConvolutionLayer m_ConvolutionLayer;

    arm_compute::ConvolutionMethod m_ConvolutionMethod;

    std::unique_ptr<ICLTensorProxy> m_InputProxy;
    std::unique_ptr<ICLTensorProxy> m_WeightsProxy;
    std::unique_ptr<ICLTensorProxy> m_BiasProxy;
    std::unique_ptr<ICLTensorProxy> m_OutputProxy;
};

} //namespace armnn

