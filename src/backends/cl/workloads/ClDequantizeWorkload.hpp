//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLDequantizationLayer.h>

namespace armnn
{

arm_compute::Status ClDequantizeWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class ClDequantizeWorkload : public ClBaseWorkload<DequantizeQueueDescriptor>
{
public:
    ClDequantizeWorkload(const DequantizeQueueDescriptor& descriptor,
                         const WorkloadInfo& workloadInfo,
                         const arm_compute::CLCompileContext& clCompileContext);

    void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::CLDequantizationLayer> m_Layer;
};

} // namespace armnn