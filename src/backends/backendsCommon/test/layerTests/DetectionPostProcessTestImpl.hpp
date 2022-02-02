//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <ResolveType.hpp>

#include <armnn/Types.hpp>

#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadFactoryHelper.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

#include <doctest/doctest.h>

namespace
{

using FloatData = std::vector<float>;
using QuantData = std::pair<float, int32_t>;

struct TestData
{
    static const armnn::TensorShape s_BoxEncodingsShape;
    static const armnn::TensorShape s_ScoresShape;
    static const armnn::TensorShape s_AnchorsShape;

    static const QuantData s_BoxEncodingsQuantData;
    static const QuantData s_ScoresQuantData;
    static const QuantData s_AnchorsQuantData;

    static const FloatData s_BoxEncodings;
    static const FloatData s_Scores;
    static const FloatData s_Anchors;
};

struct RegularNmsExpectedResults
{
    static const FloatData s_DetectionBoxes;
    static const FloatData s_DetectionScores;
    static const FloatData s_DetectionClasses;
    static const FloatData s_NumDetections;
};

struct FastNmsExpectedResults
{
    static const FloatData s_DetectionBoxes;
    static const FloatData s_DetectionScores;
    static const FloatData s_DetectionClasses;
    static const FloatData s_NumDetections;
};

const armnn::TensorShape TestData::s_BoxEncodingsShape = { 1, 6, 4 };
const armnn::TensorShape TestData::s_ScoresShape       = { 1, 6, 3 };
const armnn::TensorShape TestData::s_AnchorsShape      = { 6, 4 };

const QuantData TestData::s_BoxEncodingsQuantData = { 1.00f, 1 };
const QuantData TestData::s_ScoresQuantData       = { 0.01f, 0 };
const QuantData TestData::s_AnchorsQuantData      = { 0.50f, 0 };

const FloatData TestData::s_BoxEncodings =
{
    0.0f,  0.0f, 0.0f, 0.0f,
    0.0f,  1.0f, 0.0f, 0.0f,
    0.0f, -1.0f, 0.0f, 0.0f,
    0.0f,  0.0f, 0.0f, 0.0f,
    0.0f,  1.0f, 0.0f, 0.0f,
    0.0f,  0.0f, 0.0f, 0.0f
};

const FloatData TestData::s_Scores =
{
    0.0f, 0.90f, 0.80f,
    0.0f, 0.75f, 0.72f,
    0.0f, 0.60f, 0.50f,
    0.0f, 0.93f, 0.95f,
    0.0f, 0.50f, 0.40f,
    0.0f, 0.30f, 0.20f
};

const FloatData TestData::s_Anchors =
{
    0.5f,   0.5f, 1.0f, 1.0f,
    0.5f,   0.5f, 1.0f, 1.0f,
    0.5f,   0.5f, 1.0f, 1.0f,
    0.5f,  10.5f, 1.0f, 1.0f,
    0.5f,  10.5f, 1.0f, 1.0f,
    0.5f, 100.5f, 1.0f, 1.0f
};

const FloatData RegularNmsExpectedResults::s_DetectionBoxes =
{
    0.0f, 10.0f, 1.0f, 11.0f,
    0.0f, 10.0f, 1.0f, 11.0f,
    0.0f,  0.0f, 0.0f,  0.0f
};

const FloatData RegularNmsExpectedResults::s_DetectionScores =
{
    0.95f, 0.93f, 0.0f
};

const FloatData RegularNmsExpectedResults::s_DetectionClasses =
{
    1.0f, 0.0f, 0.0f
};

const FloatData RegularNmsExpectedResults::s_NumDetections = { 2.0f };

const FloatData FastNmsExpectedResults::s_DetectionBoxes =
{
    0.0f,  10.0f, 1.0f,  11.0f,
    0.0f,   0.0f, 1.0f,   1.0f,
    0.0f, 100.0f, 1.0f, 101.0f
};

const FloatData FastNmsExpectedResults::s_DetectionScores =
{
    0.95f, 0.9f, 0.3f
};

const FloatData FastNmsExpectedResults::s_DetectionClasses =
{
    1.0f, 0.0f, 0.0f
};

const FloatData FastNmsExpectedResults::s_NumDetections = { 3.0f };

} // anonymous namespace

template<typename FactoryType,
         armnn::DataType ArmnnType,
         typename T = armnn::ResolveType<ArmnnType>>
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
    std::unique_ptr<armnn::IProfiler> profiler = std::make_unique<armnn::IProfiler>();
    armnn::ProfilerManager::GetInstance().RegisterProfiler(profiler.get());

    auto memoryManager = WorkloadFactoryHelper<FactoryType>::GetMemoryManager();
    FactoryType workloadFactory = WorkloadFactoryHelper<FactoryType>::GetFactory(memoryManager);
    auto tensorHandleFactory = WorkloadFactoryHelper<FactoryType>::GetTensorHandleFactory(memoryManager);

    armnn::TensorInfo detectionBoxesInfo({ 1, 3, 4 }, armnn::DataType::Float32);
    armnn::TensorInfo detectionClassesInfo({ 1, 3 }, armnn::DataType::Float32);
    armnn::TensorInfo detectionScoresInfo({ 1, 3 }, armnn::DataType::Float32);
    armnn::TensorInfo numDetectionInfo({ 1 }, armnn::DataType::Float32);

    std::vector<float> actualDetectionBoxesOutput(detectionBoxesInfo.GetNumElements());
    std::vector<float> actualDetectionClassesOutput(detectionClassesInfo.GetNumElements());
    std::vector<float> actualDetectionScoresOutput(detectionScoresInfo.GetNumElements());
    std::vector<float> actualNumDetectionOutput(numDetectionInfo.GetNumElements());

    auto boxedHandle = tensorHandleFactory.CreateTensorHandle(boxEncodingsInfo);
    auto scoreshandle = tensorHandleFactory.CreateTensorHandle(scoresInfo);
    auto anchorsHandle = tensorHandleFactory.CreateTensorHandle(anchorsInfo);
    auto outputBoxesHandle = tensorHandleFactory.CreateTensorHandle(detectionBoxesInfo);
    auto classesHandle = tensorHandleFactory.CreateTensorHandle(detectionClassesInfo);
    auto outputScoresHandle = tensorHandleFactory.CreateTensorHandle(detectionScoresInfo);
    auto numDetectionHandle = tensorHandleFactory.CreateTensorHandle(numDetectionInfo);

    armnn::ScopedTensorHandle anchorsTensor(anchorsInfo);
    AllocateAndCopyDataToITensorHandle(&anchorsTensor, anchorsData.data());

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
    AddInputToWorkload(data,  info, scoresInfo, scoreshandle.get());
    AddOutputToWorkload(data, info, detectionBoxesInfo, outputBoxesHandle.get());
    AddOutputToWorkload(data, info, detectionClassesInfo, classesHandle.get());
    AddOutputToWorkload(data, info, detectionScoresInfo, outputScoresHandle.get());
    AddOutputToWorkload(data, info, numDetectionInfo, numDetectionHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::DetectionPostProcess,
                                                                                data,
                                                                                info);

    boxedHandle->Allocate();
    scoreshandle->Allocate();
    outputBoxesHandle->Allocate();
    classesHandle->Allocate();
    outputScoresHandle->Allocate();
    numDetectionHandle->Allocate();

    CopyDataToITensorHandle(boxedHandle.get(), boxEncodingsData.data());
    CopyDataToITensorHandle(scoreshandle.get(), scoresData.data());

    workload->Execute();

    CopyDataFromITensorHandle(actualDetectionBoxesOutput.data(), outputBoxesHandle.get());
    CopyDataFromITensorHandle(actualDetectionClassesOutput.data(), classesHandle.get());
    CopyDataFromITensorHandle(actualDetectionScoresOutput.data(), outputScoresHandle.get());
    CopyDataFromITensorHandle(actualNumDetectionOutput.data(), numDetectionHandle.get());

    auto result = CompareTensors(actualDetectionBoxesOutput,
                                 expectedDetectionBoxes,
                                 outputBoxesHandle->GetShape(),
                                 detectionBoxesInfo.GetShape());
    CHECK_MESSAGE(result.m_Result, result.m_Message.str());

    result = CompareTensors(actualDetectionClassesOutput,
                            expectedDetectionClasses,
                            classesHandle->GetShape(),
                            detectionClassesInfo.GetShape());
    CHECK_MESSAGE(result.m_Result, result.m_Message.str());

    result = CompareTensors(actualDetectionScoresOutput,
                            expectedDetectionScores,
                            outputScoresHandle->GetShape(),
                            detectionScoresInfo.GetShape());
    CHECK_MESSAGE(result.m_Result, result.m_Message.str());

    result = CompareTensors(actualNumDetectionOutput,
                            expectedNumDetections,
                            numDetectionHandle->GetShape(),
                            numDetectionInfo.GetShape());
    CHECK_MESSAGE(result.m_Result, result.m_Message.str());
}

template<armnn::DataType QuantizedType, typename RawType = armnn::ResolveType<QuantizedType>>
void QuantizeData(RawType* quant, const float* dequant, const armnn::TensorInfo& info)
{
    for (size_t i = 0; i < info.GetNumElements(); i++)
    {
        quant[i] = armnn::Quantize<RawType>(
            dequant[i], info.GetQuantizationScale(), info.GetQuantizationOffset());
    }
}

template<typename FactoryType>
void DetectionPostProcessRegularNmsFloatTest()
{
    return DetectionPostProcessImpl<FactoryType, armnn::DataType::Float32>(
        armnn::TensorInfo(TestData::s_BoxEncodingsShape, armnn::DataType::Float32),
        armnn::TensorInfo(TestData::s_ScoresShape, armnn::DataType::Float32),
        armnn::TensorInfo(TestData::s_AnchorsShape, armnn::DataType::Float32),
        TestData::s_BoxEncodings,
        TestData::s_Scores,
        TestData::s_Anchors,
        RegularNmsExpectedResults::s_DetectionBoxes,
        RegularNmsExpectedResults::s_DetectionClasses,
        RegularNmsExpectedResults::s_DetectionScores,
        RegularNmsExpectedResults::s_NumDetections,
        true);
}

template<typename FactoryType,
         armnn::DataType QuantizedType,
         typename RawType = armnn::ResolveType<QuantizedType>>
void DetectionPostProcessRegularNmsQuantizedTest()
{
    armnn::TensorInfo boxEncodingsInfo(TestData::s_BoxEncodingsShape, QuantizedType);
    armnn::TensorInfo scoresInfo(TestData::s_ScoresShape, QuantizedType);
    armnn::TensorInfo anchorsInfo(TestData::s_AnchorsShape, QuantizedType);

    boxEncodingsInfo.SetQuantizationScale(TestData::s_BoxEncodingsQuantData.first);
    boxEncodingsInfo.SetQuantizationOffset(TestData::s_BoxEncodingsQuantData.second);

    scoresInfo.SetQuantizationScale(TestData::s_ScoresQuantData.first);
    scoresInfo.SetQuantizationOffset(TestData::s_ScoresQuantData.second);

    anchorsInfo.SetQuantizationScale(TestData::s_AnchorsQuantData.first);
    anchorsInfo.SetQuantizationOffset(TestData::s_BoxEncodingsQuantData.second);

    std::vector<RawType> boxEncodingsData(TestData::s_BoxEncodingsShape.GetNumElements());
    QuantizeData<QuantizedType>(boxEncodingsData.data(),
                                TestData::s_BoxEncodings.data(),
                                boxEncodingsInfo);

    std::vector<RawType> scoresData(TestData::s_ScoresShape.GetNumElements());
    QuantizeData<QuantizedType>(scoresData.data(),
                                TestData::s_Scores.data(),
                                scoresInfo);

    std::vector<RawType> anchorsData(TestData::s_AnchorsShape.GetNumElements());
    QuantizeData<QuantizedType>(anchorsData.data(),
                                TestData::s_Anchors.data(),
                                anchorsInfo);

    return DetectionPostProcessImpl<FactoryType, QuantizedType>(
        boxEncodingsInfo,
        scoresInfo,
        anchorsInfo,
        boxEncodingsData,
        scoresData,
        anchorsData,
        RegularNmsExpectedResults::s_DetectionBoxes,
        RegularNmsExpectedResults::s_DetectionClasses,
        RegularNmsExpectedResults::s_DetectionScores,
        RegularNmsExpectedResults::s_NumDetections,
        true);
}

template<typename FactoryType>
void DetectionPostProcessFastNmsFloatTest()
{
    return DetectionPostProcessImpl<FactoryType, armnn::DataType::Float32>(
        armnn::TensorInfo(TestData::s_BoxEncodingsShape, armnn::DataType::Float32),
        armnn::TensorInfo(TestData::s_ScoresShape, armnn::DataType::Float32),
        armnn::TensorInfo(TestData::s_AnchorsShape, armnn::DataType::Float32),
        TestData::s_BoxEncodings,
        TestData::s_Scores,
        TestData::s_Anchors,
        FastNmsExpectedResults::s_DetectionBoxes,
        FastNmsExpectedResults::s_DetectionClasses,
        FastNmsExpectedResults::s_DetectionScores,
        FastNmsExpectedResults::s_NumDetections,
        false);
}

template<typename FactoryType,
         armnn::DataType QuantizedType,
         typename RawType = armnn::ResolveType<QuantizedType>>
void DetectionPostProcessFastNmsQuantizedTest()
{
    armnn::TensorInfo boxEncodingsInfo(TestData::s_BoxEncodingsShape, QuantizedType);
    armnn::TensorInfo scoresInfo(TestData::s_ScoresShape, QuantizedType);
    armnn::TensorInfo anchorsInfo(TestData::s_AnchorsShape, QuantizedType);

    boxEncodingsInfo.SetQuantizationScale(TestData::s_BoxEncodingsQuantData.first);
    boxEncodingsInfo.SetQuantizationOffset(TestData::s_BoxEncodingsQuantData.second);

    scoresInfo.SetQuantizationScale(TestData::s_ScoresQuantData.first);
    scoresInfo.SetQuantizationOffset(TestData::s_ScoresQuantData.second);

    anchorsInfo.SetQuantizationScale(TestData::s_AnchorsQuantData.first);
    anchorsInfo.SetQuantizationOffset(TestData::s_BoxEncodingsQuantData.second);

    std::vector<RawType> boxEncodingsData(TestData::s_BoxEncodingsShape.GetNumElements());
    QuantizeData<QuantizedType>(boxEncodingsData.data(),
                                TestData::s_BoxEncodings.data(),
                                boxEncodingsInfo);

    std::vector<RawType> scoresData(TestData::s_ScoresShape.GetNumElements());
    QuantizeData<QuantizedType>(scoresData.data(),
                                TestData::s_Scores.data(),
                                scoresInfo);

    std::vector<RawType> anchorsData(TestData::s_AnchorsShape.GetNumElements());
    QuantizeData<QuantizedType>(anchorsData.data(),
                                TestData::s_Anchors.data(),
                                anchorsInfo);

    return DetectionPostProcessImpl<FactoryType, QuantizedType>(
        boxEncodingsInfo,
        scoresInfo,
        anchorsInfo,
        boxEncodingsData,
        scoresData,
        anchorsData,
        FastNmsExpectedResults::s_DetectionBoxes,
        FastNmsExpectedResults::s_DetectionClasses,
        FastNmsExpectedResults::s_DetectionScores,
        FastNmsExpectedResults::s_NumDetections,
        false);
}
