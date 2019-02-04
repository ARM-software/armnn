//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "reference/workloads/DetectionPostProcess.cpp"

#include <armnn/Descriptors.hpp>
#include <armnn/Types.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(RefDetectionPostProcess)


BOOST_AUTO_TEST_CASE(TopKSortTest)
{
    unsigned int k = 3;
    unsigned int indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    float values[8] = { 0, 7, 6, 5, 4, 3, 2, 500 };
    TopKSort(k, indices, values, 8);
    BOOST_TEST(indices[0] == 7);
    BOOST_TEST(indices[1] == 1);
    BOOST_TEST(indices[2] == 2);
}

BOOST_AUTO_TEST_CASE(FullTopKSortTest)
{
    unsigned int k = 8;
    unsigned int indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    float values[8] = { 0, 7, 6, 5, 4, 3, 2, 500 };
    TopKSort(k, indices, values, 8);
    BOOST_TEST(indices[0] == 7);
    BOOST_TEST(indices[1] == 1);
    BOOST_TEST(indices[2] == 2);
    BOOST_TEST(indices[3] == 3);
    BOOST_TEST(indices[4] == 4);
    BOOST_TEST(indices[5] == 5);
    BOOST_TEST(indices[6] == 6);
    BOOST_TEST(indices[7] == 0);
}

BOOST_AUTO_TEST_CASE(IouTest)
{
    float boxI[4] = { 0.0f, 0.0f, 10.0f, 10.0f };
    float boxJ[4] = { 1.0f, 1.0f, 11.0f, 11.0f };
    float iou = IntersectionOverUnion(boxI, boxJ);
    BOOST_TEST(iou == 0.68, boost::test_tools::tolerance(0.001));
}

BOOST_AUTO_TEST_CASE(NmsFunction)
{
    std::vector<float> boxCorners({
        0.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 0.1f, 1.0f, 1.1f,
        0.0f, -0.1f, 1.0f, 0.9f,
        0.0f, 10.0f, 1.0f, 11.0f,
        0.0f, 10.1f, 1.0f, 11.1f,
        0.0f, 100.0f, 1.0f, 101.0f
    });

    std::vector<float> scores({ 0.9f, 0.75f, 0.6f, 0.93f, 0.5f, 0.3f });

    std::vector<unsigned int> result = NonMaxSuppression(6, boxCorners, scores, 0.0, 3, 0.5);
    BOOST_TEST(result.size() == 3);
    BOOST_TEST(result[0] == 3);
    BOOST_TEST(result[1] == 0);
    BOOST_TEST(result[2] == 5);
}

void DetectionPostProcessTestImpl(bool useRegularNms, const std::vector<float>& expectedDetectionBoxes,
                                  const std::vector<float>& expectedDetectionClasses,
                                  const std::vector<float>& expectedDetectionScores,
                                  const std::vector<float>& expectedNumDetections)
{
    armnn::TensorInfo boxEncodingsInfo({ 1, 6, 4 }, armnn::DataType::Float32);
    armnn::TensorInfo scoresInfo({ 1, 6, 3 }, armnn::DataType::Float32);
    armnn::TensorInfo anchorsInfo({ 6, 4 }, armnn::DataType::Float32);

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

    std::vector<float> detectionBoxes(detectionBoxesInfo.GetNumElements());
    std::vector<float> detectionScores(detectionScoresInfo.GetNumElements());
    std::vector<float> detectionClasses(detectionClassesInfo.GetNumElements());
    std::vector<float> numDetections(1);

    armnn::DetectionPostProcess(boxEncodingsInfo, scoresInfo, anchorsInfo,
                                detectionBoxesInfo, detectionClassesInfo,
                                detectionScoresInfo, numDetectionInfo, desc,
                                boxEncodings.data(), scores.data(), anchors.data(),
                                detectionBoxes.data(), detectionClasses.data(),
                                detectionScores.data(), numDetections.data());

    BOOST_TEST(detectionBoxes == expectedDetectionBoxes);
    BOOST_TEST(detectionScores == expectedDetectionScores);
    BOOST_TEST(detectionClasses == expectedDetectionClasses);
    BOOST_TEST(numDetections == expectedNumDetections);
}

BOOST_AUTO_TEST_CASE(RegularNmsDetectionPostProcess)
{
    std::vector<float> expectedDetectionBoxes({
        0.0f, 10.0f, 1.0f, 11.0f,
        0.0f, 10.0f, 1.0f, 11.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    });

    std::vector<float> expectedDetectionScores({ 0.95f, 0.93f, 0.0f });
    std::vector<float> expectedDetectionClasses({ 1.0f, 0.0f, 0.0f });
    std::vector<float> expectedNumDetections({ 2.0f });

    DetectionPostProcessTestImpl(true, expectedDetectionBoxes, expectedDetectionClasses,
                                 expectedDetectionScores, expectedNumDetections);
}

BOOST_AUTO_TEST_CASE(FastNmsDetectionPostProcess)
{
    std::vector<float> expectedDetectionBoxes({
        0.0f, 10.0f, 1.0f, 11.0f,
        0.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 100.0f, 1.0f, 101.0f
    });
    std::vector<float> expectedDetectionScores({ 0.95f, 0.9f, 0.3f });
    std::vector<float> expectedDetectionClasses({ 1.0f, 0.0f, 0.0f });
    std::vector<float> expectedNumDetections({ 3.0f });

    DetectionPostProcessTestImpl(false, expectedDetectionBoxes, expectedDetectionClasses,
                                 expectedDetectionScores, expectedNumDetections);
}

BOOST_AUTO_TEST_SUITE_END()