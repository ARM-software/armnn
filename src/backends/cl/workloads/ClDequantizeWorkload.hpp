//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLDequantizationLayer.h>

namespace armnn
{

arm_compute::Status ClDequantizeWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class ClDequantizeWorkload : public BaseWorkload<DequantizeQueueDescriptor>
{
public:
    ClDequantizeWorkload(const DequantizeQueueDescriptor& descriptor, const WorkloadInfo& workloadInfo);

    void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::CLDequantizationLayer> m_Layer;
};

} // namespace armnn