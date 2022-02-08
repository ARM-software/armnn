//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>
#include <arm_compute/runtime/Tensor.h>

#include <memory>

namespace armnn
{

arm_compute::Status NeonFullyConnectedWorkloadValidate(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const TensorInfo& weights,
                                                       const Optional<TensorInfo>& biases,
                                                       const FullyConnectedDescriptor& descriptor,
                                                       const ActivationDescriptor* activationDescriptor = nullptr);

class NeonFullyConnectedWorkload : public NeonBaseWorkload<FullyConnectedQueueDescriptor>
{
public:
    NeonFullyConnectedWorkload(const FullyConnectedQueueDescriptor& descriptor, const WorkloadInfo& info,
                               std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);
    virtual void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_FullyConnectedLayer;

    std::unique_ptr<arm_compute::Tensor> m_WeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_BiasesTensor;

    void FreeUnusedTensors();
};

} //namespace armnn

