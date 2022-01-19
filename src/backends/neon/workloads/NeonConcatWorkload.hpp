//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/NEON/functions/NEConcatenateLayer.h>

#include <memory>

namespace armnn
{
arm_compute::Status NeonConcatWorkloadValidate(const std::vector<const TensorInfo*>& inputs,
                                               const TensorInfo& output,
                                               const OriginsDescriptor& descriptor);

class NeonConcatWorkload : public NeonBaseWorkload<ConcatQueueDescriptor>
{
public:
    NeonConcatWorkload(const ConcatQueueDescriptor& descriptor, const WorkloadInfo& info);

    using NeonBaseWorkload<ConcatQueueDescriptor>::NeonBaseWorkload;
    void Execute() const override;

private:
    std::unique_ptr<arm_compute::NEConcatenateLayer> m_Layer;
};

} //namespace armnn
