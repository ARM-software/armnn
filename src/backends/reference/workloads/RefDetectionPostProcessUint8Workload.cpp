//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefDetectionPostProcessUint8Workload.hpp"

#include "DetectionPostProcess.hpp"
#include "Profiling.hpp"
#include "RefWorkloadUtils.hpp"

namespace armnn
{

RefDetectionPostProcessUint8Workload::RefDetectionPostProcessUint8Workload(
        const DetectionPostProcessQueueDescriptor& descriptor, const WorkloadInfo& info)
        : Uint8ToFloat32Workload<DetectionPostProcessQueueDescriptor>(descriptor, info),
          m_Anchors(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Anchors))) {}

void RefDetectionPostProcessUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDetectionPostProcessUint8Workload_Execute");

    const TensorInfo& boxEncodingsInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& scoresInfo = GetTensorInfo(m_Data.m_Inputs[1]);
    const TensorInfo& anchorsInfo = GetTensorInfo(m_Anchors.get());
    const TensorInfo& detectionBoxesInfo = GetTensorInfo(m_Data.m_Outputs[0]);
    const TensorInfo& detectionClassesInfo = GetTensorInfo(m_Data.m_Outputs[1]);
    const TensorInfo& detectionScoresInfo = GetTensorInfo(m_Data.m_Outputs[2]);
    const TensorInfo& numDetectionsInfo = GetTensorInfo(m_Data.m_Outputs[3]);

    const uint8_t* boxEncodingsData = GetInputTensorDataU8(0, m_Data);
    const uint8_t* scoresData = GetInputTensorDataU8(1, m_Data);
    const uint8_t* anchorsData = m_Anchors->GetConstTensor<uint8_t>();

    auto boxEncodings = Dequantize(boxEncodingsData, boxEncodingsInfo);
    auto scores = Dequantize(scoresData, scoresInfo);
    auto anchors = Dequantize(anchorsData, anchorsInfo);

    float* detectionBoxes = GetOutputTensorData<float>(0, m_Data);
    float* detectionClasses = GetOutputTensorData<float>(1, m_Data);
    float* detectionScores = GetOutputTensorData<float>(2, m_Data);
    float* numDetections = GetOutputTensorData<float>(3, m_Data);

    DetectionPostProcess(boxEncodingsInfo, scoresInfo, anchorsInfo,
                         detectionBoxesInfo, detectionClassesInfo,
                         detectionScoresInfo, numDetectionsInfo, m_Data.m_Parameters,
                         boxEncodings.data(), scores.data(), anchors.data(),
                         detectionBoxes, detectionClasses, detectionScores, numDetections);
}

} //namespace armnn
