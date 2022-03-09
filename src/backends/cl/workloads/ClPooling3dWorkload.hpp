//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLPooling3dLayer.h>

namespace armnn
{

    arm_compute::Status ClPooling3dWorkloadValidate(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    const Pooling3dDescriptor& descriptor);

    class ClPooling3dWorkload : public ClBaseWorkload<Pooling3dQueueDescriptor>
    {
    public:
        using BaseWorkload<Pooling3dQueueDescriptor>::m_Data;

        ClPooling3dWorkload(const Pooling3dQueueDescriptor& descriptor,
                            const WorkloadInfo& info,
                            const arm_compute::CLCompileContext& clCompileContext);

        void Execute() const override;

    private:
        mutable arm_compute::CLPooling3dLayer m_PoolingLayer;
    };

} //namespace armnn
