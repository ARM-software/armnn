//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "TensorCopyUtils.hpp"
#include "TypeUtils.hpp"
#include "WorkloadTestUtils.hpp"

#include <armnn/Types.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/WorkloadFactory.hpp>
#include <backendsCommon/test/WorkloadFactoryHelper.hpp>
#include <test/TensorHelpers.hpp>

template <typename FactoryType, armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void DetectionPostProcessImpl(const armnn::TensorInfo& boxEncodingsInfo,
                              const armnn::TensorInfo& scoresInfo,
                              const armnn::TensorInfo& anchorsInfo,
                              const std::vector<T>& boxEncodingsData,
                              const std::vector<T>& scoresData,
                              const std::vector<T>& anchorsData,
                              const std::vector<float>& expectedDetectionBoxes,
                              const std::vector<float>& expectedDetectionClasses,
                              const std::vector<float>& expectedDetectionScores,
                              const std::vector<float>& expectedNumDetections,
                              bool useRegularNms)
{
    std::unique_ptr<armnn::Profiler> profiler = std::make_unique<armnn::Profiler>();
    armnn::ProfilerManager::GetInstance().RegisterProfiler(profiler.get());

    auto memoryManager = WorkloadFactoryHelper<FactoryType>::GetMemoryManager();
    FactoryType workloadFactory = WorkloadFactoryHelper<FactoryType>::GetFactory(memoryManager);

    auto boxEncodings = MakeTensor<T, 3>(boxEncodingsInfo, boxEncodingsData);
    auto scores = MakeTensor<T, 3>(scoresInfo, scoresData);
    auto anchors = MakeTensor<T, 2>(anchorsInfo, anchorsData);

    armnn::TensorInfo detectionBoxesInfo({ 1, 3, 4 }, armnn::DataType::Float32);
    armnn::TensorInfo detectionScoresInfo({ 1, 3 }, armnn::DataType::Float32);
    armnn::TensorInfo detectionClassesInfo({ 1, 3 }, armnn::DataType::Float32);
    armnn::TensorInfo numDetectionInfo({ 1 }, armnn::DataType::Float32);

    LayerTestResult<float, 3> detectionBoxesResult(detectionBoxesInfo);
    detectionBoxesResult.outputExpected = MakeTensor<float, 3>(detectionBoxesInfo, expectedDetectionBoxes);
    LayerTestResult<float, 2> detectionClassesResult(detectionClassesInfo);
    detectionClassesResult.outputExpected = MakeTensor<float, 2>(detectionClassesInfo, expectedDetectionClasses);
    LayerTestResult<float, 2> detectionScoresResult(detectionScoresInfo);
    detectionScoresResult.outputExpected = MakeTensor<float, 2>(detectionScoresInfo, expectedDetectionScores);
    LayerTestResult<float, 1> numDetectionsResult(numDetectionInfo);
    numDetectionsResult.outputExpected = MakeTensor<float, 1>(numDetectionInfo, expectedNumDetections);

    std::unique_ptr<armnn::ITensorHandle> boxedHandle = workloadFactory.CreateTensorHandle(boxEncodingsInfo);
    std::unique_ptr<armnn::ITensorHandle> scoreshandle = workloadFactory.CreateTensorHandle(scoresInfo);
    std::unique_ptr<armnn::ITensorHandle> anchorsHandle = workloadFactory.CreateTensorHandle(anchorsInfo);
    std::unique_ptr<armnn::ITensorHandle> outputBoxesHandle = workloadFactory.CreateTensorHandle(detectionBoxesInfo);
    std::unique_ptr<armnn::ITensorHandle> classesHandle = workloadFactory.CreateTensorHandle(detectionClassesInfo);
    std::unique_ptr<armnn::ITensorHandle> outputScoresHandle = workloadFactory.CreateTensorHandle(detectionScoresInfo);
    std::unique_ptr<armnn::ITensorHandle> numDetectionHandle = workloadFactory.CreateTensorHandle(numDetectionInfo);

    armnn::ScopedCpuTensorHandle anchorsTensor(anchorsInfo);
    AllocateAndCopyDataToITensorHandle(&anchorsTensor, &anchors[0][0]);

    armnn::DetectionPostProcessQueueDescriptor data;
    data.m_Parameters.m_UseRegularNms = useRegularNms;
    data.m_Parameters.m_MaxDetections = 3;
    data.m_Parameters.m_MaxClassesPerDetection = 1;
    data.m_Parameters.m_DetectionsPerClass =1;
    data.m_Parameters.m_NmsScoreThreshold = 0.0;
    data.m_Parameters.m_NmsIouThreshold = 0.5;
    data.m_Parameters.m_NumClasses = 2;
    data.m_Parameters.m_ScaleY = 10.0;
    data.m_Parameters.m_ScaleX = 10.0;
    data.m_Parameters.m_ScaleH = 5.0;
    data.m_Parameters.m_ScaleW = 5.0;
    data.m_Anchors = &anchorsTensor;

    armnn::WorkloadInfo info;
    AddInputToWorkload(data,  info, boxEncodingsInfo, boxedHandle.get());
    AddInputToWorkload(data, info, scoresInfo, scoreshandle.get());
    AddOutputToWorkload(data, info, detectionBoxesInfo, outputBoxesHandle.get());
    AddOutputToWorkload(data, info, detectionClassesInfo, classesHandle.get());
    AddOutputToWorkload(data, info, detectionScoresInfo, outputScoresHandle.get());
    AddOutputToWorkload(data, info, numDetectionInfo, numDetectionHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateDetectionPostProcess(data, info);

    boxedHandle->Allocate();
    scoreshandle->Allocate();
    outputBoxesHandle->Allocate();
    classesHandle->Allocate();
    outputScoresHandle->Allocate();
    numDetectionHandle->Allocate();

    CopyDataToITensorHandle(boxedHandle.get(), boxEncodings.origin());
    CopyDataToITensorHandle(scoreshandle.get(), scores.origin());

    workload->Execute();

    CopyDataFromITensorHandle(detectionBoxesResult.output.origin(), outputBoxesHandle.get());
    CopyDataFromITensorHandle(detectionClassesResult.output.origin(), classesHandle.get());
    CopyDataFromITensorHandle(detectionScoresResult.output.origin(), outputScoresHandle.get());
    CopyDataFromITensorHandle(numDetectionsResult.output.origin(), numDetectionHandle.get());

    BOOST_TEST(CompareTensors(detectionBoxesResult.output, detectionBoxesResult.outputExpected));
    BOOST_TEST(CompareTensors(detectionClassesResult.output, detectionClassesResult.outputExpected));
    BOOST_TEST(CompareTensors(detectionScoresResult.output, detectionScoresResult.outputExpected));
    BOOST_TEST(CompareTensors(numDetectionsResult.output, numDetectionsResult.outputExpected));
}

inline void QuantizeData(uint8_t* quant, const float* dequant, const armnn::TensorInfo& info)
{
    for (size_t i = 0; i < info.GetNumElements(); i++)
    {
        quant[i] = armnn::Quantize<uint8_t>(dequant[i], info.GetQuantizationScale(), info.GetQuantizationOffset());
    }
}

template <typename FactoryType>
void DetectionPostProcessRegularNmsFloatTest()
{
    armnn::TensorInfo boxEncodingsInfo({ 1, 6, 4 }, armnn::DataType::Float32);
    armnn::TensorInfo scoresInfo({ 1, 6, 3}, armnn::DataType::Float32);
    armnn::TensorInfo anchorsInfo({ 6, 4 }, armnn::DataType::Float32);

    std::vector<float> boxEncodingsData({
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    });
    std::vector<float> scoresData({
        0.0f, 0.9f, 0.8f,
        0.0f, 0.75f, 0.72f,
        0.0f, 0.6f, 0.5f,
        0.0f, 0.93f, 0.95f,
        0.0f, 0.5f, 0.4f,
        0.0f, 0.3f, 0.2f
    });
    std::vector<float> anchorsData({
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 100.5f, 1.0f, 1.0f
    });

    std::vector<float> expectedDetectionBoxes({
        0.0f, 10.0f, 1.0f, 11.0f,
        0.0f, 10.0f, 1.0f, 11.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    });
    std::vector<float> expectedDetectionScores({ 0.95f, 0.93f, 0.0f });
    std::vector<float> expectedDetectionClasses({ 1.0f, 0.0f, 0.0f });
    std::vector<float> expectedNumDetections({ 2.0f });

    return DetectionPostProcessImpl<FactoryType, armnn::DataType::Float32>(boxEncodingsInfo,
                                                                           scoresInfo,
                                                                           anchorsInfo,
                                                                           boxEncodingsData,
                                                                           scoresData,
                                                                           anchorsData,
                                                                           expectedDetectionBoxes,
                                                                           expectedDetectionClasses,
                                                                           expectedDetectionScores,
                                                                           expectedNumDetections,
                                                                           true);
}

template <typename FactoryType>
void DetectionPostProcessRegularNmsUint8Test()
{
    armnn::TensorInfo boxEncodingsInfo({ 1, 6, 4 }, armnn::DataType::QuantisedAsymm8);
    armnn::TensorInfo scoresInfo({ 1, 6, 3 }, armnn::DataType::QuantisedAsymm8);
    armnn::TensorInfo anchorsInfo({ 6, 4 }, armnn::DataType::QuantisedAsymm8);

    boxEncodingsInfo.SetQuantizationScale(1.0f);
    boxEncodingsInfo.SetQuantizationOffset(1);
    scoresInfo.SetQuantizationScale(0.01f);
    scoresInfo.SetQuantizationOffset(0);
    anchorsInfo.SetQuantizationScale(0.5f);
    anchorsInfo.SetQuantizationOffset(0);

    std::vector<float> boxEncodings({
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    });
    std::vector<float> scores({
        0.0f, 0.9f, 0.8f,
        0.0f, 0.75f, 0.72f,
        0.0f, 0.6f, 0.5f,
        0.0f, 0.93f, 0.95f,
        0.0f, 0.5f, 0.4f,
        0.0f, 0.3f, 0.2f
    });
    std::vector<float> anchors({
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 100.5f, 1.0f, 1.0f
    });

    std::vector<uint8_t> boxEncodingsData(boxEncodings.size(), 0);
    std::vector<uint8_t> scoresData(scores.size(), 0);
    std::vector<uint8_t> anchorsData(anchors.size(), 0);
    QuantizeData(boxEncodingsData.data(), boxEncodings.data(), boxEncodingsInfo);
    QuantizeData(scoresData.data(), scores.data(), scoresInfo);
    QuantizeData(anchorsData.data(), anchors.data(), anchorsInfo);

    std::vector<float> expectedDetectionBoxes({
        0.0f, 10.0f, 1.0f, 11.0f,
        0.0f, 10.0f, 1.0f, 11.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    });
    std::vector<float> expectedDetectionScores({ 0.95f, 0.93f, 0.0f });
    std::vector<float> expectedDetectionClasses({ 1.0f, 0.0f, 0.0f });
    std::vector<float> expectedNumDetections({ 2.0f });

    return DetectionPostProcessImpl<FactoryType, armnn::DataType::QuantisedAsymm8>(boxEncodingsInfo,
                                                                                   scoresInfo,
                                                                                   anchorsInfo,
                                                                                   boxEncodingsData,
                                                                                   scoresData,
                                                                                   anchorsData,
                                                                                   expectedDetectionBoxes,
                                                                                   expectedDetectionClasses,
                                                                                   expectedDetectionScores,
                                                                                   expectedNumDetections,
                                                                                   true);
}

template <typename FactoryType>
void DetectionPostProcessFastNmsFloatTest()
{
    armnn::TensorInfo boxEncodingsInfo({ 1, 6, 4 }, armnn::DataType::Float32);
    armnn::TensorInfo scoresInfo({ 1, 6, 3}, armnn::DataType::Float32);
    armnn::TensorInfo anchorsInfo({ 6, 4 }, armnn::DataType::Float32);

    std::vector<float> boxEncodingsData({
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    });
    std::vector<float> scoresData({
        0.0f, 0.9f, 0.8f,
        0.0f, 0.75f, 0.72f,
        0.0f, 0.6f, 0.5f,
        0.0f, 0.93f, 0.95f,
        0.0f, 0.5f, 0.4f,
        0.0f, 0.3f, 0.2f
    });
    std::vector<float> anchorsData({
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 100.5f, 1.0f, 1.0f
    });

    std::vector<float> expectedDetectionBoxes({
        0.0f, 10.0f, 1.0f, 11.0f,
        0.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 100.0f, 1.0f, 101.0f
    });
    std::vector<float> expectedDetectionScores({ 0.95f, 0.9f, 0.3f });
    std::vector<float> expectedDetectionClasses({ 1.0f, 0.0f, 0.0f });
    std::vector<float> expectedNumDetections({ 3.0f });

    return DetectionPostProcessImpl<FactoryType, armnn::DataType::Float32>(boxEncodingsInfo,
                                                                           scoresInfo,
                                                                           anchorsInfo,
                                                                           boxEncodingsData,
                                                                           scoresData,
                                                                           anchorsData,
                                                                           expectedDetectionBoxes,
                                                                           expectedDetectionClasses,
                                                                           expectedDetectionScores,
                                                                           expectedNumDetections,
                                                                           false);
}

template <typename FactoryType>
void DetectionPostProcessFastNmsUint8Test()
{
    armnn::TensorInfo boxEncodingsInfo({ 1, 6, 4 }, armnn::DataType::QuantisedAsymm8);
    armnn::TensorInfo scoresInfo({ 1, 6, 3 }, armnn::DataType::QuantisedAsymm8);
    armnn::TensorInfo anchorsInfo({ 6, 4 }, armnn::DataType::QuantisedAsymm8);

    boxEncodingsInfo.SetQuantizationScale(1.0f);
    boxEncodingsInfo.SetQuantizationOffset(1);
    scoresInfo.SetQuantizationScale(0.01f);
    scoresInfo.SetQuantizationOffset(0);
    anchorsInfo.SetQuantizationScale(0.5f);
    anchorsInfo.SetQuantizationOffset(0);

    std::vector<float> boxEncodings({
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    });
    std::vector<float> scores({
        0.0f, 0.9f, 0.8f,
        0.0f, 0.75f, 0.72f,
        0.0f, 0.6f, 0.5f,
        0.0f, 0.93f, 0.95f,
        0.0f, 0.5f, 0.4f,
        0.0f, 0.3f, 0.2f
    });
    std::vector<float> anchors({
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 10.5f, 1.0f, 1.0f,
        0.5f, 100.5f, 1.0f, 1.0f
    });

    std::vector<uint8_t> boxEncodingsData(boxEncodings.size(), 0);
    std::vector<uint8_t> scoresData(scores.size(), 0);
    std::vector<uint8_t> anchorsData(anchors.size(), 0);
    QuantizeData(boxEncodingsData.data(), boxEncodings.data(), boxEncodingsInfo);
    QuantizeData(scoresData.data(), scores.data(), scoresInfo);
    QuantizeData(anchorsData.data(), anchors.data(), anchorsInfo);

    std::vector<float> expectedDetectionBoxes({
        0.0f, 10.0f, 1.0f, 11.0f,
        0.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 100.0f, 1.0f, 101.0f
    });
    std::vector<float> expectedDetectionScores({ 0.95f, 0.9f, 0.3f });
    std::vector<float> expectedDetectionClasses({ 1.0f, 0.0f, 0.0f });
    std::vector<float> expectedNumDetections({ 3.0f });

    return DetectionPostProcessImpl<FactoryType, armnn::DataType::QuantisedAsymm8>(boxEncodingsInfo,
                                                                                   scoresInfo,
                                                                                   anchorsInfo,
                                                                                   boxEncodingsData,
                                                                                   scoresData,
                                                                                   anchorsData,
                                                                                   expectedDetectionBoxes,
                                                                                   expectedDetectionClasses,
                                                                                   expectedDetectionScores,
                                                                                   expectedNumDetections,
                                                                                   false);
}
