//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/Tensor.h>

#include <memory>

namespace armnn
{
    arm_compute::Status NeonBatchMatMulValidate(const TensorInfo& inputX,
                                                const TensorInfo& inputY,
                                                const TensorInfo& output,
                                                const BatchMatMulDescriptor& descriptor);

    class NeonBatchMatMulWorkload : public NeonBaseWorkload<BatchMatMulQueueDescriptor>
    {
    public:
        NeonBatchMatMulWorkload(const BatchMatMulQueueDescriptor& descriptor,
                                const WorkloadInfo& info);
        virtual void Execute() const override;

    private:
        // ACL layers required to fully form a Batch Mat Mul layer.
        std::unique_ptr<arm_compute::IFunction> m_GEMMLayer;
        std::unique_ptr<arm_compute::IFunction> m_PermuteLayerX;
        std::unique_ptr<arm_compute::IFunction> m_PermuteLayerY;

        // Additional ACL arm_compute::Tensors.
        // Required to perform permutations.
        arm_compute::Tensor m_PermutedTensorX;
        arm_compute::Tensor m_PermutedTensorY;

    };
} //namespace armnn
