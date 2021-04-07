//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ImageUtils.hpp"
#include "BoundingBox.hpp"
#include "Types.hpp"

#include <armnn/Logging.hpp>

static cv::Scalar GetScalarColorCode(std::tuple<int, int, int> color)
{
    return cv::Scalar(std::get<0>(color), std::get<1>(color), std::get<2>(color));
}

void AddInferenceOutputToFrame(od::DetectedObjects& decodedResults, cv::Mat& inputFrame,
                               std::vector<std::tuple<std::string, common::BBoxColor>>& labels)
{
    for(const od::DetectedObject& object : decodedResults)
    {
        int confidence = static_cast<int>(object.GetScore() * 100);
        int baseline = 0;
        std::string textStr;
        std::tuple<int, int, int> colorCode(255, 0, 0); //red

        if (labels.size() > object.GetId())
        {
            auto label = labels[object.GetId()];
            textStr = std::get<0>(label) + " - " + std::to_string(confidence) + "%";
            colorCode = std::get<1>(label).colorCode;
        }
        else
        {
            textStr = std::to_string(object.GetId()) + " - " + std::to_string(confidence) + "%";
        }

        cv::Size textSize = getTextSize(textStr, cv::FONT_HERSHEY_DUPLEX, 1.0, 1, &baseline);

        const od::BoundingBox& bbox = object.GetBoundingBox();

        if (bbox.GetX() + bbox.GetWidth() > inputFrame.cols)
        {
            cv::Rect r(bbox.GetX(), bbox.GetY(), inputFrame.cols - bbox.GetX(), bbox.GetHeight());

            cv::rectangle(inputFrame, r, GetScalarColorCode(colorCode), 2, 8, 0);
        }
        else if (bbox.GetY() + bbox.GetHeight() > inputFrame.rows)
        {
            cv::Rect r(bbox.GetX(), bbox.GetY(), bbox.GetWidth(), inputFrame.rows - bbox.GetY());

            cv::rectangle(inputFrame, r, GetScalarColorCode(colorCode), 2, 8, 0);
        }
        else
        {
            cv::Rect r(bbox.GetX(), bbox.GetY(), bbox.GetWidth(), bbox.GetHeight());

            cv::rectangle(inputFrame, r, GetScalarColorCode(colorCode), 2, 8, 0);
        }

        int textBoxY = std::max(0 ,bbox.GetY() - textSize.height);

        cv::Rect text(bbox.GetX(), textBoxY, textSize.width, textSize.height);

        cv::rectangle(inputFrame, text, GetScalarColorCode(colorCode), -1);

        cv::Scalar color;

        if(std::get<0>(colorCode) + std::get<1>(colorCode) + std::get<2>(colorCode) > 127)
        {
            color = cv::Scalar::all(0);
        }
        else
        {
            color = cv::Scalar::all(255);
        }

        cv::putText(inputFrame,
                    textStr ,
                    cv::Point(bbox.GetX(), textBoxY + textSize.height -(textSize.height)/3),
                    cv::FONT_HERSHEY_DUPLEX,
                    0.5,
                    color,
                    1);
    }
}


void ResizeFrame(const cv::Mat& frame, cv::Mat& dest, const common::Size& aspectRatio)
{
    if(&dest != &frame)
    {
        double longEdgeInput = std::max(frame.rows, frame.cols);
        double longEdgeOutput = std::max(aspectRatio.m_Width, aspectRatio.m_Height);
        const double resizeFactor = longEdgeOutput/longEdgeInput;
        cv::resize(frame, dest, cv::Size(0, 0), resizeFactor, resizeFactor, DefaultResizeFlag);
    }
    else
    {
        const std::string warningMessage{"Resize was not performed because resized frame references the source frame."};
        ARMNN_LOG(warning) << warningMessage;
    }
}

/** Pad a frame with zeros (add rows and columns to the end) */
void PadFrame(const cv::Mat& src, cv::Mat& dest, const int bottom, const int right)
{
    if(&dest != &src)
    {
        cv::copyMakeBorder(src, dest, 0, bottom, 0, right, cv::BORDER_CONSTANT);
    }
    else
    {
        const std::string warningMessage
        {
            "Pad was not performed because destination frame references the source frame."
        };
        ARMNN_LOG(warning) << warningMessage;
    }
}

void ResizeWithPad(const cv::Mat& frame, cv::Mat& dest, cv::Mat& cache, const common::Size& destSize)
{
    ResizeFrame(frame, cache, destSize);
    PadFrame(cache, dest,destSize.m_Height - cache.rows,destSize.m_Width - cache.cols);
}
