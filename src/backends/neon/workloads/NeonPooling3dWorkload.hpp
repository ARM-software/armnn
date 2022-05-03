//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/runtime/NEON/functions/NEPooling3dLayer.h>

namespace armnn
{

    arm_compute::Status NeonPooling3dWorkloadValidate(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const Pooling3dDescriptor& descriptor);

    class NeonPooling3dWorkload : public NeonBaseWorkload<Pooling3dQueueDescriptor>
    {
    public:
        using BaseWorkload<Pooling3dQueueDescriptor>::m_Data;

        NeonPooling3dWorkload(const Pooling3dQueueDescriptor& descriptor,
                            const WorkloadInfo& info);

        void Execute() const override;

    private:
        std::unique_ptr<arm_compute::IFunction> m_PoolingLayer;
    };

} //namespace armnn
