//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

#include "arm_compute/runtime/MemoryManagerOnDemand.h"

#include <memory>

namespace armnn
{

arm_compute::Status ClFullyConnectedWorkloadValidate(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const TensorInfo& weights,
                                                     const TensorInfo& biases,
                                                     const FullyConnectedDescriptor& descriptor);

class ClFullyConnectedFloatWorkload : public armnn::FloatWorkload<armnn::FullyConnectedQueueDescriptor>
{
public:
    ClFullyConnectedFloatWorkload(const armnn::FullyConnectedQueueDescriptor& descriptor,
                                  const armnn::WorkloadInfo& info,
                                  std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);

    using armnn::FloatWorkload<armnn::FullyConnectedQueueDescriptor>::m_Data;
    void Execute() const override;

private:
    mutable arm_compute::CLFullyConnectedLayer m_FullyConnectedLayer;

    std::unique_ptr<arm_compute::CLTensor> m_WeightsTensor;
    std::unique_ptr<arm_compute::CLTensor> m_BiasesTensor;

    void FreeUnusedTensors();
};

} //namespace armnn
