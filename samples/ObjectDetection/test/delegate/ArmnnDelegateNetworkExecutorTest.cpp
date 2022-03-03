//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <catch.hpp>
#include <opencv2/opencv.hpp>
#include "ArmnnNetworkExecutor.hpp"
#include "Types.hpp"
#include "ImageUtils.hpp"
#include "SSDResultDecoder.hpp"
#include "YoloResultDecoder.hpp"

using namespace std;

static string GetResourceFilePath(const string& filename)
{
    string testResources = TEST_RESOURCE_DIR;

    if(testResources.back() != '/')
    {
        return testResources + "/" + filename;
    }
    else
    {
        return testResources + filename;
    }
}

TEST_CASE("Test Delegate Execution SSD_MOBILE")
{
    string testResources = TEST_RESOURCE_DIR;
    REQUIRE(testResources != "");

    vector<armnn::BackendId> m_backends = {"CpuRef"};
    string file_path = GetResourceFilePath("ssd_mobilenet_v1.tflite");
    common::InferenceResults<float> results;
    cv::Mat processed;
    cv::Mat cache;
    float detectionThreshold = 0.6;
    common::Profiling profiling(true);

    profiling.ProfilingStart();
    auto executor = make_unique<common::ArmnnNetworkExecutor<float>>(file_path, m_backends, true);
    int width = executor->GetImageAspectRatio().m_Width;
    int height = executor->GetImageAspectRatio().m_Height;
    od::SSDResultDecoder ssdResult(detectionThreshold);

    /* check GetInputDataType */
    CHECK(executor->GetInputDataType() == armnn::DataType::QAsymmU8);
    /* check GetImageAspectRatio */
    CHECK(width == 300);
    CHECK(height == 300);

    cv::Mat inputFrame = cv::imread(GetResourceFilePath("basketball1.png"), cv::IMREAD_COLOR);
    cv::cvtColor(inputFrame, inputFrame, cv::COLOR_BGR2RGB);
    ResizeWithPad(inputFrame, processed, cache, common::Size(width,height));
    CHECK(executor->Run(processed.data, processed.total() * processed.elemSize(), results) == true);
    od::DetectedObjects detections = ssdResult.Decode(results,
                      common::Size(inputFrame.size().width, inputFrame.size().height),
                      common::Size(width, height), {});

    /* Make sure we've found 2 persons in the image */
    CHECK(detections.size() == 2 );
    CHECK(detections[0].GetLabel() == "0");
    CHECK(detections[1].GetLabel() == "0");
    /* check GetQuantizationScale */
    CHECK(to_string(executor->GetQuantizationScale()) == string("0.007812"));
    /* check GetQuantizationOffset */
    CHECK(executor->GetQuantizationOffset() == 128);
    /* check GetQuantizationScale */
    CHECK(executor->GetOutputQuantizationScale(0) == 0.0f);
    /* check GetOutputQuantizationOffset */
    CHECK(executor->GetOutputQuantizationOffset(0) == 0);
    profiling.ProfilingStopAndPrintUs("Overall test");
}

TEST_CASE("Test Delegate Execution YOLO_V3")
{
    string testResources = TEST_RESOURCE_DIR;
    REQUIRE(testResources != "");

    vector<armnn::BackendId> m_backends = {"CpuRef"};
    string file_path = GetResourceFilePath("yolo_v3_tiny_darknet_fp32.tflite");
    common::InferenceResults<float> results;
    cv::Mat processed;
    cv::Mat cache;
    float NMSThreshold = 0.3f;
    float ClsThreshold = 0.3f;
    float ObjectThreshold = 0.3f;


    auto executor = make_unique<common::ArmnnNetworkExecutor<float>>(file_path, m_backends);
    int width = executor->GetImageAspectRatio().m_Width;
    int height = executor->GetImageAspectRatio().m_Height;
    od::YoloResultDecoder yoloResult(NMSThreshold, ClsThreshold, ObjectThreshold);

    /* check GetInputDataType */
    CHECK(executor->GetInputDataType() == armnn::DataType::Float32);
    /* check GetImageAspectRatio */
    CHECK(width == 416);
    CHECK(height == 416);

    /* read the image */
    cv::Mat inputFrame = cv::imread(GetResourceFilePath("basketball1.png"), cv::IMREAD_COLOR);
    /* resize it according to the the input tensor requirments */
    ResizeWithPad(inputFrame, processed, cache, common::Size(width,height));
    /* converting to 3 channel matrix of 32 bits floats */
    processed.convertTo(processed, CV_32FC3);
    /* run the inference */
    CHECK(executor->Run(processed.data, processed.total() * processed.elemSize(), results) == true);
    /* decode the results */
    od::DetectedObjects detections = yoloResult.Decode(results,
                      common::Size(inputFrame.size().width, inputFrame.size().height),
                      common::Size(width, height), {});

    /* Make sure we've found 2 persons in the image */
    CHECK(detections.size() == 2 );
    CHECK(detections[0].GetLabel() == "0");
    CHECK(detections[1].GetLabel() == "0");
    /* check GetQuantizationScale */
    CHECK(to_string(executor->GetQuantizationScale()) == string("0.000000"));
    /* check GetQuantizationOffset */
    CHECK(executor->GetQuantizationOffset() == 0);
    /* check GetQuantizationScale */
    CHECK(executor->GetOutputQuantizationScale(0) == 0.0f);
    /* check GetOutputQuantizationOffset */
    CHECK(executor->GetOutputQuantizationOffset(0) == 0);

}
