//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//


#include "NMS.hpp"

#include <cmath>
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <ostream>

namespace yolov3 {
namespace {
/** Number of elements needed to represent a box */
constexpr int box_elements = 4;
/** Number of elements needed to represent a confidence factor */
constexpr int confidence_elements = 1;

/** Calculate Intersection Over Union of two boxes
 *
 * @param[in] box1 First box
 * @param[in] box2 Second box
 *
 * @return The IoU of the two boxes
 */
float iou(const Box& box1, const Box& box2)
{
    const float area1 = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin);
    const float area2 = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin);
    float overlap;
    if (area1 <= 0 || area2 <= 0)
    {
        overlap = 0.0f;
    }
    else
    {
        const auto y_min_intersection = std::max<float>(box1.ymin, box2.ymin);
        const auto x_min_intersection = std::max<float>(box1.xmin, box2.xmin);
        const auto y_max_intersection = std::min<float>(box1.ymax, box2.ymax);
        const auto x_max_intersection = std::min<float>(box1.xmax, box2.xmax);
        const auto area_intersection =
            std::max<float>(y_max_intersection - y_min_intersection, 0.0f) *
            std::max<float>(x_max_intersection - x_min_intersection, 0.0f);
        overlap = area_intersection / (area1 + area2 - area_intersection);
    }
    return overlap;
}

std::vector<Detection> convert_to_detections(const NMSConfig& config,
                                             const std::vector<float>& detected_boxes)
{
    const size_t element_step = static_cast<size_t>(
        box_elements + confidence_elements + config.num_classes);
    std::vector<Detection> detections;

    for (unsigned int i = 0; i < config.num_boxes; ++i)
    {
        const float* cur_box = &detected_boxes[i * element_step];
        if (cur_box[4] > config.confidence_threshold)
        {
            Detection det;
            det.box = {cur_box[0], cur_box[0] + cur_box[2], cur_box[1],
                       cur_box[1] + cur_box[3]};
            det.confidence = cur_box[4];
            det.classes.resize(static_cast<size_t>(config.num_classes), 0);
            for (unsigned int c = 0; c < config.num_classes; ++c)
            {
                const float class_prob = det.confidence * cur_box[5 + c];
                if (class_prob > config.confidence_threshold)
                {
                    det.classes[c] = class_prob;
                }
            }
            detections.emplace_back(std::move(det));
        }
    }
    return detections;
}
} // namespace

bool compare_detection(const yolov3::Detection& detection,
                       const std::vector<float>& expected)
{
    float tolerance = 0.001f;
    return (std::fabs(detection.classes[0] - expected[0]) < tolerance  &&
            std::fabs(detection.box.xmin   - expected[1]) < tolerance  &&
            std::fabs(detection.box.ymin   - expected[2]) < tolerance  &&
            std::fabs(detection.box.xmax   - expected[3]) < tolerance  &&
            std::fabs(detection.box.ymax   - expected[4]) < tolerance  &&
            std::fabs(detection.confidence - expected[5]) < tolerance  );
}

void print_detection(std::ostream& os,
                     const std::vector<Detection>& detections)
{
    for (const auto& detection : detections)
    {
        for (unsigned int c = 0; c < detection.classes.size(); ++c)
        {
            if (detection.classes[c] != 0.0f)
            {
                os << c << " " << detection.classes[c] << " " << detection.box.xmin
                   << " " << detection.box.ymin << " " << detection.box.xmax << " "
                   << detection.box.ymax << std::endl;
            }
        }
    }
}

std::vector<Detection> nms(const NMSConfig& config,
                           const std::vector<float>& detected_boxes) {
    // Get detections that comply with the expected confidence threshold
    std::vector<Detection> detections =
        convert_to_detections(config, detected_boxes);

    const unsigned int num_detections = static_cast<unsigned int>(detections.size());
    for (unsigned int c = 0; c < config.num_classes; ++c)
    {
        // Sort classes
        std::sort(detections.begin(), detections.begin() + static_cast<std::ptrdiff_t>(num_detections),
                  [c](Detection& detection1, Detection& detection2)
                    {
                        return (detection1.classes[c] - detection2.classes[c]) > 0;
                    });
        // Clear detections with high IoU
        for (unsigned int d = 0; d < num_detections; ++d)
        {
            // Check if class is already cleared/invalidated
            if (detections[d].classes[c] == 0.f)
            {
                continue;
            }

            // Filter out boxes on IoU threshold
            const Box& box1 = detections[d].box;
            for (unsigned int b = d + 1; b < num_detections; ++b)
            {
                const Box& box2 = detections[b].box;
                if (iou(box1, box2) > config.iou_threshold)
                {
                    detections[b].classes[c] = 0.f;
                }
            }
        }
    }
    return detections;
}
} // namespace yolov3
