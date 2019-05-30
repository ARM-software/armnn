//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/NEON/NEFunctions.h>

#include <functional>

namespace armnn
{

arm_compute::Status NeonDequantizeWorkloadValidate(const TensorInfo& input,
                                                   const TensorInfo& output);

class NeonDequantizeWorkload : public BaseWorkload<DequantizeQueueDescriptor>
{
public:
    NeonDequantizeWorkload(const DequantizeQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::NEDequantizationLayer> m_Layer;
};

} //namespace armnn
