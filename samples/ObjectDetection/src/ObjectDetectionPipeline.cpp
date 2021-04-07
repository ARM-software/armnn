//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ObjectDetectionPipeline.hpp"
#include "ImageUtils.hpp"

namespace od
{

ObjDetectionPipeline::ObjDetectionPipeline(std::unique_ptr<common::ArmnnNetworkExecutor<float>> executor,
                                           std::unique_ptr<IDetectionResultDecoder> decoder) :
        m_executor(std::move(executor)),
        m_decoder(std::move(decoder)){}

void od::ObjDetectionPipeline::Inference(const cv::Mat& processed, common::InferenceResults<float>& result)
{
    m_executor->Run(processed.data, processed.total() * processed.elemSize(), result);
}

void ObjDetectionPipeline::PostProcessing(common::InferenceResults<float>& inferenceResult,
        const std::function<void (DetectedObjects)>& callback)
{
    DetectedObjects detections = m_decoder->Decode(inferenceResult, m_inputImageSize,
                                           m_executor->GetImageAspectRatio(), {});
    if (callback)
    {
        callback(detections);
    }
}

void ObjDetectionPipeline::PreProcessing(const cv::Mat& frame, cv::Mat& processed)
{
    m_inputImageSize.m_Height = frame.rows;
    m_inputImageSize.m_Width = frame.cols;
    ResizeWithPad(frame, processed, m_processedFrame, m_executor->GetImageAspectRatio());
}

MobileNetSSDv1::MobileNetSSDv1(std::unique_ptr<common::ArmnnNetworkExecutor<float>> executor,
                               float objectThreshold) :
        ObjDetectionPipeline(std::move(executor),
                             std::make_unique<SSDResultDecoder>(objectThreshold))
{}

void MobileNetSSDv1::PreProcessing(const cv::Mat& frame, cv::Mat& processed)
{
    ObjDetectionPipeline::PreProcessing(frame, processed);
    if (m_executor->GetInputDataType() == armnn::DataType::Float32)
    {
        // [0, 255] => [-1.0, 1.0]
        processed.convertTo(processed, CV_32FC3, 1 / 127.5, -1);
    }
}

YoloV3Tiny::YoloV3Tiny(std::unique_ptr<common::ArmnnNetworkExecutor<float>> executor,
                       float NMSThreshold, float ClsThreshold, float ObjectThreshold) :
        ObjDetectionPipeline(std::move(executor),
                             std::move(std::make_unique<YoloResultDecoder>(NMSThreshold,
                                                                           ClsThreshold,
                                                                           ObjectThreshold)))
{}

void YoloV3Tiny::PreProcessing(const cv::Mat& frame, cv::Mat& processed)
{
    ObjDetectionPipeline::PreProcessing(frame, processed);
    if (m_executor->GetInputDataType() == armnn::DataType::Float32)
    {
        processed.convertTo(processed, CV_32FC3);
    }
}

IPipelinePtr CreatePipeline(common::PipelineOptions& config)
{
    auto executor = std::make_unique<common::ArmnnNetworkExecutor<float>>(config.m_ModelFilePath, config.m_backends);

    if (config.m_ModelName == "SSD_MOBILE")
    {
        float detectionThreshold = 0.6;

        return std::make_unique<od::MobileNetSSDv1>(std::move(executor),
                                                    detectionThreshold
        );
    }
    else if (config.m_ModelName == "YOLO_V3_TINY")
    {
        float NMSThreshold = 0.6f;
        float ClsThreshold = 0.6f;
        float ObjectThreshold = 0.6f;
        return std::make_unique<od::YoloV3Tiny>(std::move(executor),
                                                NMSThreshold,
                                                ClsThreshold,
                                                ObjectThreshold
        );
    }
    else
    {
        throw std::invalid_argument("Unknown Model name: " + config.m_ModelName + " supplied by user.");
    }

}
}// namespace od