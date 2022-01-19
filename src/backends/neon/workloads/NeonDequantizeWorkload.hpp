//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/IFunction.h>

#include <functional>

namespace armnn
{

arm_compute::Status NeonDequantizeWorkloadValidate(const TensorInfo& input,
                                                   const TensorInfo& output);

class NeonDequantizeWorkload : public NeonBaseWorkload<DequantizeQueueDescriptor>
{
public:
    NeonDequantizeWorkload(const DequantizeQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::IFunction> m_Layer;
};

} //namespace armnn
