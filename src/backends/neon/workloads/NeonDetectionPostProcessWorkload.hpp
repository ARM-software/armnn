//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/NEON/functions/NEDetectionPostProcessLayer.h>

namespace armnn
{

arm_compute::Status NeonDetectionPostProcessValidate(const TensorInfo& boxEncodings,
                                                     const TensorInfo& scores,
                                                     const TensorInfo& anchors,
                                                     const TensorInfo& detectionBoxes,
                                                     const TensorInfo& detectionClasses,
                                                     const TensorInfo& detectionScores,
                                                     const TensorInfo& numDetections,
                                                     const DetectionPostProcessDescriptor &desc);

class NeonDetectionPostProcessWorkload : public BaseWorkload<DetectionPostProcessQueueDescriptor>
{
public:
    NeonDetectionPostProcessWorkload(
        const DetectionPostProcessQueueDescriptor& descriptor,
        const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEDetectionPostProcessLayer m_Func;

    std::unique_ptr<arm_compute::Tensor> m_Anchors;

};

} // namespace armnn