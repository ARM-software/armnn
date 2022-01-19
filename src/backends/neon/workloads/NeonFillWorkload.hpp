//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/WorkloadData.hpp>
#include "NeonBaseWorkload.hpp"
#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/core/Error.h>

namespace armnn {

class NeonFillWorkload : public NeonBaseWorkload<FillQueueDescriptor>
{
public:
    NeonFillWorkload(const FillQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_Layer;
};

} //namespace armnn
