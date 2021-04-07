//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ArmnnNetworkExecutor.hpp"
#include "YoloResultDecoder.hpp"
#include "SSDResultDecoder.hpp"
# include "ImageUtils.hpp"

#include <opencv2/opencv.hpp>

namespace od
{
/**
 * Generic object detection pipeline with 3 steps: data pre-processing, inference execution and inference
 * result post-processing.
 *
 */
class ObjDetectionPipeline {
public:

    /**
     * Creates object detection pipeline with given network executor and decoder.
     * @param executor - unique pointer to inference runner
     * @param decoder - unique pointer to inference results decoder
     */
    ObjDetectionPipeline(std::unique_ptr<common::ArmnnNetworkExecutor<float>> executor,
                         std::unique_ptr<IDetectionResultDecoder> decoder);

    /**
     * @brief Standard image pre-processing implementation.
     *
     * Re-sizes an image keeping aspect ratio, pads if necessary to fit the network input layer dimensions.

     * @param[in] frame - input image, expected data type is uint8.
     * @param[out] processed - output image, data type is preserved.
     */
    virtual void PreProcessing(const cv::Mat& frame, cv::Mat& processed);

    /**
     * @brief Executes inference
     *
     * Calls inference runner provided during instance construction.
     *
     * @param[in] processed - input inference data. Data type should be aligned with input tensor.
     * @param[out] result - raw floating point inference results.
     */
    virtual void Inference(const cv::Mat& processed, common::InferenceResults<float>& result);

    /**
     * @brief Standard inference results post-processing implementation.
     *
     * Decodes inference results using decoder provided during construction.
     *
     * @param[in] inferenceResult - inference results to be decoded.
     * @param[in] callback - a function to be called after successful inference results decoding.
     */
    virtual void PostProcessing(common::InferenceResults<float>& inferenceResult,
                                const std::function<void (DetectedObjects)>& callback);

protected:
    std::unique_ptr<common::ArmnnNetworkExecutor<float>> m_executor;
    std::unique_ptr<IDetectionResultDecoder> m_decoder;
    common::Size m_inputImageSize{};
    cv::Mat m_processedFrame;
};

/**
 * Specific to Yolo v3 tiny object detection pipeline implementation.
 */
class YoloV3Tiny: public ObjDetectionPipeline{
public:

    /**
     * Constructs object detection pipeline for Yolo v3 tiny network.
     *
     * Network input is expected to be uint8 or fp32. Data range [0, 255].
     * Network output is FP32.
     *
     * @param executor[in] - unique pointer to inference runner
     * @param NMSThreshold[in] - non max suppression threshold for decoding step
     * @param ClsThreshold[in] -  class probability threshold for decoding step
     * @param ObjectThreshold[in] - detected object score threshold for decoding step
     */
    YoloV3Tiny(std::unique_ptr<common::ArmnnNetworkExecutor<float>> executor,
               float NMSThreshold, float ClsThreshold, float ObjectThreshold);

    /**
     * @brief Yolo v3 tiny image pre-processing implementation.
     *
     * On top of the standard pre-processing, converts input data type according to the network input tensor data type.
     * Supported data types: uint8 and float32.
     *
     * @param[in] original - input image data
     * @param[out] processed - image data ready to be used for inference.
     */
    void PreProcessing(const cv::Mat& original, cv::Mat& processed);

};

/**
 * Specific to MobileNet SSD v1 object detection pipeline implementation.
 */
class MobileNetSSDv1: public ObjDetectionPipeline {

public:
    /**
     * Constructs object detection pipeline for MobileNet SSD network.
     *
     * Network input is expected to be uint8 or fp32. Data range [-1, 1].
     * Network output is FP32.
     *
     * @param[in] - unique pointer to inference runner
     * @paramp[in] objectThreshold - detected object score threshold for decoding step
     */
    MobileNetSSDv1(std::unique_ptr<common::ArmnnNetworkExecutor<float>> executor,
                   float objectThreshold);

    /**
     * @brief MobileNet SSD image pre-processing implementation.
     *
     * On top of the standard pre-processing, converts input data type according to the network input tensor data type
     * and scales input data from [0, 255] to [-1, 1] for FP32 input.
     *
     * Supported input data types: uint8 and float32.
     *
     * @param[in] original - input image data
     * @param processed[out] - image data ready to be used for inference.
     */
    void PreProcessing(const cv::Mat& original, cv::Mat& processed);

};

using IPipelinePtr = std::unique_ptr<od::ObjDetectionPipeline>;

/**
 * Constructs object detection pipeline based on configuration provided.
 *
 * @param[in] config - object detection pipeline configuration.
 *
 * @return unique pointer to object detection pipeline.
 */
IPipelinePtr CreatePipeline(common::PipelineOptions& config);

}// namespace od