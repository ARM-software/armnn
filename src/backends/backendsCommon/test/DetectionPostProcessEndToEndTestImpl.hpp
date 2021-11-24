//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <CommonTestUtils.hpp>

#include <armnn/INetwork.hpp>
#include <ResolveType.hpp>

#include <doctest/doctest.h>

namespace{

template<typename T>
armnn::INetworkPtr CreateDetectionPostProcessNetwork(const armnn::TensorInfo& boxEncodingsInfo,
                                                     const armnn::TensorInfo& scoresInfo,
                                                     const armnn::TensorInfo& anchorsInfo,
                                                     const std::vector<T>& anchors,
                                                     bool useRegularNms)
{
    armnn::TensorInfo detectionBoxesInfo({ 1, 3, 4 }, armnn::DataType::Float32);
    armnn::TensorInfo detectionScoresInfo({ 1, 3 }, armnn::DataType::Float32);
    armnn::TensorInfo detectionClassesInfo({ 1, 3 }, armnn::DataType::Float32);
    armnn::TensorInfo numDetectionInfo({ 1 }, armnn::DataType::Float32);

    armnn::DetectionPostProcessDescriptor desc;
    desc.m_UseRegularNms = useRegularNms;
    desc.m_MaxDetections = 3;
    desc.m_MaxClassesPerDetection = 1;
    desc.m_DetectionsPerClass =1;
    desc.m_NmsScoreThreshold = 0.0;
    desc.m_NmsIouThreshold = 0.5;
    desc.m_NumClasses = 2;
    desc.m_ScaleY = 10.0;
    desc.m_ScaleX = 10.0;
    desc.m_ScaleH = 5.0;
    desc.m_ScaleW = 5.0;

    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* boxesLayer = net->AddInputLayer(0);
    armnn::IConnectableLayer* scoresLayer = net->AddInputLayer(1);
    armnn::ConstTensor anchorsTensor(anchorsInfo, anchors.data());
    armnn::IConnectableLayer* detectionLayer = net->AddDetectionPostProcessLayer(desc, anchorsTensor,
                                                                                 "DetectionPostProcess");
    armnn::IConnectableLayer* detectionBoxesLayer = net->AddOutputLayer(0, "detectionBoxes");
    armnn::IConnectableLayer* detectionClassesLayer = net->AddOutputLayer(1, "detectionClasses");
    armnn::IConnectableLayer* detectionScoresLayer = net->AddOutputLayer(2, "detectionScores");
    armnn::IConnectableLayer* numDetectionLayer = net->AddOutputLayer(3, "numDetection");
    Connect(boxesLayer, detectionLayer, boxEncodingsInfo, 0, 0);
    Connect(scoresLayer, detectionLayer, scoresInfo, 0, 1);
    Connect(detectionLayer, detectionBoxesLayer, detectionBoxesInfo, 0, 0);
    Connect(detectionLayer, detectionClassesLayer, detectionClassesInfo, 1, 0);
    Connect(detectionLayer, detectionScoresLayer, detectionScoresInfo, 2, 0);
    Connect(detectionLayer, numDetectionLayer, numDetectionInfo, 3, 0);

    return net;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void DetectionPostProcessEndToEnd(const std::vector<BackendId>& backends, bool useRegularNms,
                                  const std::vector<T>& boxEncodings,
                                  const std::vector<T>& scores,
                                  const std::vector<T>& anchors,
                                  const std::vector<float>& expectedDetectionBoxes,
                                  const std::vector<float>& expectedDetectionClasses,
                                  const std::vector<float>& expectedDetectionScores,
                                  const std::vector<float>& expectedNumDetections,
                                  float boxScale = 1.0f,
                                  int32_t boxOffset = 0,
                                  float scoreScale = 1.0f,
                                  int32_t scoreOffset = 0,
                                  float anchorScale = 1.0f,
                                  int32_t anchorOffset = 0)
{
    armnn::TensorInfo boxEncodingsInfo({ 1, 6, 4 }, ArmnnType);
    armnn::TensorInfo scoresInfo({ 1, 6, 3}, ArmnnType);
    armnn::TensorInfo anchorsInfo({ 6, 4 }, ArmnnType);

    boxEncodingsInfo.SetQuantizationScale(boxScale);
    boxEncodingsInfo.SetQuantizationOffset(boxOffset);
    boxEncodingsInfo.SetConstant(true);
    scoresInfo.SetQuantizationScale(scoreScale);
    scoresInfo.SetQuantizationOffset(scoreOffset);
    scoresInfo.SetConstant(true);
    anchorsInfo.SetQuantizationScale(anchorScale);
    anchorsInfo.SetQuantizationOffset(anchorOffset);
    anchorsInfo.SetConstant(true);

    // Builds up the structure of the network
    armnn::INetworkPtr net = CreateDetectionPostProcessNetwork<T>(boxEncodingsInfo, scoresInfo,
                                                                  anchorsInfo, anchors, useRegularNms);

    CHECK(net);

    std::map<int, std::vector<T>> inputTensorData = {{ 0, boxEncodings }, { 1, scores }};
    std::map<int, std::vector<float>> expectedOutputData = {{ 0, expectedDetectionBoxes },
                                                            { 1, expectedDetectionClasses },
                                                            { 2, expectedDetectionScores },
                                                            { 3, expectedNumDetections }};

    EndToEndLayerTestImpl<ArmnnType, armnn::DataType::Float32>(
        move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void DetectionPostProcessRegularNmsEndToEnd(const std::vector<BackendId>& backends,
                                            const std::vector<T>& boxEncodings,
                                            const std::vector<T>& scores,
                                            const std::vector<T>& anchors,
                                            float boxScale = 1.0f,
                                            int32_t boxOffset = 0,
                                            float scoreScale = 1.0f,
                                            int32_t scoreOffset = 0,
                                            float anchorScale = 1.0f,
                                            int32_t anchorOffset = 0)
{
    std::vector<float> expectedDetectionBoxes({
        0.0f, 10.0f, 1.0f, 11.0f,
        0.0f, 10.0f, 1.0f, 11.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    });
    std::vector<float> expectedDetectionScores({ 0.95f, 0.93f, 0.0f });
    std::vector<float> expectedDetectionClasses({ 1.0f, 0.0f, 0.0f });
    std::vector<float> expectedNumDetections({ 2.0f });

    DetectionPostProcessEndToEnd<ArmnnType>(backends, true, boxEncodings, scores, anchors,
                                            expectedDetectionBoxes, expectedDetectionClasses,
                                            expectedDetectionScores, expectedNumDetections,
                                            boxScale, boxOffset, scoreScale, scoreOffset,
                                            anchorScale, anchorOffset);

};


template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void DetectionPostProcessFastNmsEndToEnd(const std::vector<BackendId>& backends,
                                         const std::vector<T>& boxEncodings,
                                         const std::vector<T>& scores,
                                         const std::vector<T>& anchors,
                                         float boxScale = 1.0f,
                                         int32_t boxOffset = 0,
                                         float scoreScale = 1.0f,
                                          int32_t scoreOffset = 0,
                                         float anchorScale = 1.0f,
                                         int32_t anchorOffset = 0)
{
    std::vector<float> expectedDetectionBoxes({
        0.0f, 10.0f, 1.0f, 11.0f,
        0.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 100.0f, 1.0f, 101.0f
    });
    std::vector<float> expectedDetectionScores({ 0.95f, 0.9f, 0.3f });
    std::vector<float> expectedDetectionClasses({ 1.0f, 0.0f, 0.0f });
    std::vector<float> expectedNumDetections({ 3.0f });

    DetectionPostProcessEndToEnd<ArmnnType>(backends, false, boxEncodings, scores, anchors,
                                            expectedDetectionBoxes, expectedDetectionClasses,
                                            expectedDetectionScores, expectedNumDetections,
                                            boxScale, boxOffset, scoreScale, scoreOffset,
                                            anchorScale, anchorOffset);

};

} // anonymous namespace
