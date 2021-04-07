//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefDetectionPostProcessWorkload.hpp"

#include "Decoders.hpp"
#include "DetectionPostProcess.hpp"
#include "Profiling.hpp"
#include "RefWorkloadUtils.hpp"

namespace armnn
{

RefDetectionPostProcessWorkload::RefDetectionPostProcessWorkload(
        const DetectionPostProcessQueueDescriptor& descriptor, const WorkloadInfo& info)
        : BaseWorkload<DetectionPostProcessQueueDescriptor>(descriptor, info),
          m_Anchors(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Anchors))) {}

void RefDetectionPostProcessWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefDetectionPostProcessWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefDetectionPostProcessWorkload::Execute(std::vector<ITensorHandle*> inputs,
                                              std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDetectionPostProcessWorkload_Execute");

    const TensorInfo& boxEncodingsInfo = GetTensorInfo(inputs[0]);
    const TensorInfo& scoresInfo       = GetTensorInfo(inputs[1]);
    const TensorInfo& anchorsInfo      = m_Anchors->GetTensorInfo();

    const TensorInfo& detectionBoxesInfo   = GetTensorInfo(outputs[0]);
    const TensorInfo& detectionClassesInfo = GetTensorInfo(outputs[1]);
    const TensorInfo& detectionScoresInfo  = GetTensorInfo(outputs[2]);
    const TensorInfo& numDetectionsInfo    = GetTensorInfo(outputs[3]);

    auto boxEncodings = MakeDecoder<float>(boxEncodingsInfo, inputs[0]->Map());
    auto scores       = MakeDecoder<float>(scoresInfo, inputs[1]->Map());
    auto anchors      = MakeDecoder<float>(anchorsInfo, m_Anchors->Map(false));

    float* detectionBoxes   = GetOutputTensorData<float>(0, m_Data);
    float* detectionClasses = GetOutputTensorData<float>(1, m_Data);
    float* detectionScores  = GetOutputTensorData<float>(2, m_Data);
    float* numDetections    = GetOutputTensorData<float>(3, m_Data);

    DetectionPostProcess(boxEncodingsInfo, scoresInfo, anchorsInfo,
                         detectionBoxesInfo, detectionClassesInfo,
                         detectionScoresInfo, numDetectionsInfo, m_Data.m_Parameters,
                         *boxEncodings, *scores, *anchors, detectionBoxes,
                         detectionClasses, detectionScores, numDetections);
}

} //namespace armnn
