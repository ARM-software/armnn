//
// Copyright © 2017-2018,2020-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"
#include "ClWorkloadUtils.hpp"

#include <arm_compute/runtime/CL/functions/CLBatchToSpaceLayer.h>
#include <arm_compute/runtime/CL/functions/CLReshapeLayer.h>

namespace armnn
{

arm_compute::Status ClBatchToSpaceNdWorkloadValidate(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const BatchToSpaceNdDescriptor& descriptor);

class ClBatchToSpaceNdWorkload : public ClBaseWorkload<BatchToSpaceNdQueueDescriptor>
{
public:
    ClBatchToSpaceNdWorkload(const BatchToSpaceNdQueueDescriptor& descriptor,
                             const WorkloadInfo& info,
                             const arm_compute::CLCompileContext& clCompileContext);

    virtual void Execute() const override;

private:
    mutable arm_compute::CLBatchToSpaceLayer m_Layer;
    mutable std::unique_ptr<arm_compute::CLReshapeLayer> m_LayerReshapeInput;
    mutable std::unique_ptr<arm_compute::CLReshapeLayer> m_LayerReshapeOutput;
    arm_compute::CLTensor m_ReshapeInputTensor;
    arm_compute::CLTensor m_ReshapeOutputTensor;
};

} //namespace armnn
