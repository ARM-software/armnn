//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ostream>
#include <vector>

namespace yolov3 {
/** Non Maxima Suprresion configuration meta-data */
struct NMSConfig {
    unsigned int num_classes{0};      /**< Number of classes in the detected boxes */
    unsigned int num_boxes{0};        /**< Number of detected boxes */
    float confidence_threshold{0.8f}; /**< Inclusion confidence threshold for a box */
    float iou_threshold{0.8f};        /**< Inclusion threshold for Intersection-Over-Union */
};

/** Box representation structure */
struct Box {
    float xmin;  /**< X-pos position of the low left coordinate */
    float xmax;  /**< X-pos position of the top right coordinate */
    float ymin;  /**< Y-pos position of the low left coordinate */
    float ymax;  /**< Y-pos position of the top right coordinate */
};

/** Detection structure */
struct Detection {
    Box box;                    /**< Detection box */
    float confidence;           /**< Confidence of detection */
    std::vector<float> classes; /**< Probability of classes */
};

/** Print identified yolo detections
 *
 * @param[in, out] os          Output stream to print to
 * @param[in]      detections  Detections to print
 */
void print_detection(std::ostream& os,
                     const std::vector<Detection>& detections);

/** Compare a detection object with a vector of float values
 *
 * @param detection [in] Detection object
 * @param expected  [in] Vector of expected float values
 * @return Boolean to represent if they match or not
 */
bool compare_detection(const yolov3::Detection& detection,
                       const std::vector<float>& expected);

/** Perform Non-Maxima Supression on a list of given detections
 *
 * @param[in] config         Configuration metadata for NMS
 * @param[in] detected_boxes Detected boxes
 *
 * @return A vector with the final detections
 */
std::vector<Detection> nms(const NMSConfig& config,
                           const std::vector<float>& detected_boxes);
} // namespace yolov3
