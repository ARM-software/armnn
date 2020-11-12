//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>

#include <memory>

namespace armnn
{

arm_compute::Status ClFullyConnectedWorkloadValidate(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const TensorInfo& weights,
                                                     const TensorInfo& biases,
                                                     const FullyConnectedDescriptor& descriptor,
                                                     const ActivationDescriptor* activationDescriptor = nullptr);

class ClFullyConnectedWorkload : public armnn::BaseWorkload<armnn::FullyConnectedQueueDescriptor>
{
public:
    ClFullyConnectedWorkload(const armnn::FullyConnectedQueueDescriptor& descriptor,
                             const armnn::WorkloadInfo& info,
                             std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);

    using armnn::BaseWorkload<armnn::FullyConnectedQueueDescriptor>::m_Data;
    void Execute() const override;

private:
    mutable arm_compute::CLFullyConnectedLayer m_FullyConnectedLayer;

    std::unique_ptr<arm_compute::CLTensor> m_WeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_BiasesTensor;

    void FreeUnusedTensors();
};

} //namespace armnn
