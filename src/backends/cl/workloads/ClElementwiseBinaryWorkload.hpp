//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLElementwiseOperations.h>

namespace armnn
{

class ClElementwiseBinaryWorkload : public ClBaseWorkload<ElementwiseBinaryQueueDescriptor>
{
public:
    ClElementwiseBinaryWorkload(const ElementwiseBinaryQueueDescriptor& descriptor,
                                const WorkloadInfo& info,
                                const arm_compute::CLCompileContext& clCompileContext);

    void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_ElementwiseBinaryLayer;

};

arm_compute::Status ClElementwiseBinaryValidate(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output,
                                                const ElementwiseBinaryDescriptor& descriptor,
                                                const ActivationDescriptor* activationDescriptor = nullptr);
} //namespace armnn