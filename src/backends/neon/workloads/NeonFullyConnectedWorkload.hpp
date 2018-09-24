//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/neon/workloads/NeonWorkloadUtils.hpp>

#include <arm_compute/runtime/MemoryManagerOnDemand.h>

#include <memory>

namespace armnn
{

arm_compute::Status NeonFullyConnectedWorkloadValidate(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const TensorInfo& weights,
                                                       const TensorInfo& biases,
                                                       const FullyConnectedDescriptor& descriptor);

class NeonFullyConnectedWorkload : public BaseWorkload<FullyConnectedQueueDescriptor>
{
public:
    NeonFullyConnectedWorkload(const FullyConnectedQueueDescriptor& descriptor, const WorkloadInfo& info,
                               std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEFullyConnectedLayer m_FullyConnectedLayer;

    std::unique_ptr<arm_compute::Tensor> m_WeightsTensor;
    std::unique_ptr<arm_compute::Tensor> m_BiasesTensor;

    void FreeUnusedTensors();
};

} //namespace armnn

