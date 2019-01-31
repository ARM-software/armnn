//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefDetectionPostProcessFloat32Workload.hpp"

#include "DetectionPostProcess.hpp"
#include "Profiling.hpp"
#include "RefWorkloadUtils.hpp"

namespace armnn
{

RefDetectionPostProcessFloat32Workload::RefDetectionPostProcessFloat32Workload(
        const DetectionPostProcessQueueDescriptor& descriptor, const WorkloadInfo& info)
        : Float32Workload<DetectionPostProcessQueueDescriptor>(descriptor, info),
          m_Anchors(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Anchors))) {}

void RefDetectionPostProcessFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDetectionPostProcessUint8Workload_Execute");

    const TensorInfo& boxEncodingsInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& scoresInfo = GetTensorInfo(m_Data.m_Inputs[1]);
    const TensorInfo& anchorsInfo = GetTensorInfo(m_Anchors.get());
    const TensorInfo& detectionBoxesInfo = GetTensorInfo(m_Data.m_Outputs[0]);
    const TensorInfo& detectionClassesInfo = GetTensorInfo(m_Data.m_Outputs[1]);
    const TensorInfo& detectionScoresInfo = GetTensorInfo(m_Data.m_Outputs[2]);
    const TensorInfo& numDetectionsInfo = GetTensorInfo(m_Data.m_Outputs[3]);

    const float* boxEncodings = GetInputTensorDataFloat(0, m_Data);
    const float* scores = GetInputTensorDataFloat(1, m_Data);
    const float* anchors = m_Anchors->GetConstTensor<float>();

    float* detectionBoxes = GetOutputTensorData<float>(0, m_Data);
    float* detectionClasses = GetOutputTensorData<float>(1, m_Data);
    float* detectionScores = GetOutputTensorData<float>(2, m_Data);
    float* numDetections = GetOutputTensorData<float>(3, m_Data);

    DetectionPostProcess(boxEncodingsInfo, scoresInfo, anchorsInfo,
                         detectionBoxesInfo, detectionClassesInfo,
                         detectionScoresInfo, numDetectionsInfo, m_Data.m_Parameters,
                         boxEncodings, scores, anchors, detectionBoxes,
                         detectionClasses, detectionScores, numDetections);
}

} //namespace armnn
