//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>

#include <memory>

namespace armnn
{

arm_compute::Status ClFullyConnectedWorkloadValidate(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const TensorInfo& weights,
                                                     const Optional<TensorInfo>& biases,
                                                     const FullyConnectedDescriptor& descriptor,
                                                     const ActivationDescriptor* activationDescriptor = nullptr);

class ClFullyConnectedWorkload : public ClBaseWorkload<FullyConnectedQueueDescriptor>
{
public:
    ClFullyConnectedWorkload(const FullyConnectedQueueDescriptor& descriptor,
                             const WorkloadInfo& info,
                             std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager,
                             const arm_compute::CLCompileContext& clCompileContext);

    using ClBaseWorkload<FullyConnectedQueueDescriptor>::m_Data;
    void Execute() const override;

private:
    mutable arm_compute::CLFullyConnectedLayer m_FullyConnectedLayer;

    std::unique_ptr<arm_compute::CLTensor> m_WeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_BiasesTensor;

    void FreeUnusedTensors();
};

} //namespace armnn
