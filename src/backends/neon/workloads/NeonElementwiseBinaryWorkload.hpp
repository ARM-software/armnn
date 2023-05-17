//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/IFunction.h>

namespace armnn
{

arm_compute::Status NeonElementwiseBinaryWorkloadValidate(const TensorInfo& input0,
                                                          const TensorInfo& input1,
                                                          const TensorInfo& output,
                                                          const ElementwiseBinaryDescriptor& descriptor,
                                                          const ActivationDescriptor* activationDescriptor = nullptr);

class NeonElementwiseBinaryWorkload : public NeonBaseWorkload<ElementwiseBinaryQueueDescriptor>
{
public:
    NeonElementwiseBinaryWorkload(const ElementwiseBinaryQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_ElementwiseBinaryLayer;
};

} //namespace armnn