//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/runtime/NEON/functions/NEQuantizationLayer.h>

namespace armnn {

arm_compute::Status NeonQuantizeWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class NeonQuantizeWorkload : public NeonBaseWorkload<QuantizeQueueDescriptor>
{
public:
    NeonQuantizeWorkload(const QuantizeQueueDescriptor& descriptor, const WorkloadInfo& workloadInfo);
    void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::NEQuantizationLayer> m_Layer;
};

} // namespace armnn
