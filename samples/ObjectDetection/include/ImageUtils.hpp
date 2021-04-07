//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "DetectedObject.hpp"
#include "Types.hpp"

#include <opencv2/opencv.hpp>

#include <vector>

const cv::InterpolationFlags DefaultResizeFlag = cv::INTER_NEAREST;

/**
* @brief Function to process the decoded results from the inference, and overlay the detail onto the provided frame
* @param[in]  decodedResults          the decoded results from the inference output.
* @param[in]  inputFrame              the frame to overlay the inference output details onto.
* @param[in]  labels                  the label set associated with the trained model used.
*/
void AddInferenceOutputToFrame(od::DetectedObjects& decodedResults,
                               cv::Mat& inputFrame,
                               std::vector<std::tuple<std::string, common::BBoxColor>>& labels);

/**
* @brief Function to resize a frame while keeping aspect ratio.
*
* @param[in]  frame            the frame we want to resize from.
* @param[out]  dest            the frame we want to resize into.
* @param[in]  aspectRatio      aspect ratio to use when resizing.
*/
void ResizeFrame(const cv::Mat& frame, cv::Mat& dest, const common::Size& aspectRatio);

/**
* @brief Function to pad a frame.
* @param[in]   src           the frame we want to pad.
* @param[out]  dest          the frame we want to store the result.
* @param[in]   bottom        padding to use on bottom of the frame.
* @param[in]   right         padding to use on the right of the frame.
*/
void PadFrame(const cv::Mat& src, cv::Mat& dest, int bottom, int right);

/**
 * Resize frame to the destination size and pad if necessary to preserve initial frame aspect ratio.
 *
 * @param frame input frame to resize
 * @param dest output frame to place resized and padded result
 * @param cache operation requires intermediate data container.
 * @param destSize size of the destination frame
 */
void ResizeWithPad(const cv::Mat& frame, cv::Mat& dest, cv::Mat& cache, const common::Size& destSize);

/**
* @brief Function to retrieve the cv::scalar color from a RGB tuple.
* @param[in]  color            the tuple form of the RGB color
*/
static cv::Scalar GetScalarColorCode(std::tuple<int, int, int> color);