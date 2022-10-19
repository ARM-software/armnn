//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/CL/CLTensor.h>
#include <memory>

namespace armnn
{
    arm_compute::Status ClBatchMatMulValidate(const TensorInfo& inputX,
                                              const TensorInfo& inputY,
                                              const TensorInfo& output,
                                              const BatchMatMulDescriptor& descriptor);

    class ClBatchMatMulWorkload : public ClBaseWorkload<BatchMatMulQueueDescriptor>
    {
    public:
        ClBatchMatMulWorkload(const BatchMatMulQueueDescriptor& descriptor,
                              const WorkloadInfo& info,
                              const arm_compute::CLCompileContext& clCompileContext);
        virtual void Execute() const override;

    private:
        // ACL layers required to fully form a Batch Mat Mul layer.
        std::unique_ptr<arm_compute::IFunction> m_GEMMLayer;
        std::unique_ptr<arm_compute::IFunction> m_PermuteLayerX;
        std::unique_ptr<arm_compute::IFunction> m_PermuteLayerY;

        // Additional CL arm_compute::Tensors.
        // Required to perform permutations.
        arm_compute::CLTensor m_PermutedTensorX;
        arm_compute::CLTensor m_PermutedTensorY;

    };
} //namespace armnn
