//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/runtime/NEON/functions/NEMatMul.h>

namespace armnn
{
    arm_compute::Status NeonBatchMatMulValidate(const TensorInfo& inputInfoX,
                                                const TensorInfo& inputInfoY,
                                                const TensorInfo& outputInfo,
                                                const BatchMatMulDescriptor& descriptor,
                                                const bool isFastMathEnabled,
                                                const ActivationDescriptor* activationDescriptor);


    class NeonBatchMatMulWorkload : public NeonBaseWorkload<BatchMatMulQueueDescriptor>
    {
    public:
        NeonBatchMatMulWorkload(const BatchMatMulQueueDescriptor& descriptor,
                                const WorkloadInfo& info,
                                const bool isFastMathEnabled);
        virtual void Execute() const override;

    private:
        mutable arm_compute::NEMatMul m_MatMulLayer;
    };
} //namespace armnn
