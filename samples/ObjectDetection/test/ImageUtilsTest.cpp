//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <catch.hpp>
#include <opencv2/opencv.hpp>
#include "ImageUtils.hpp"
#include "Types.hpp"

std::vector<std::tuple<int, int>> GetBoundingBoxPoints(std::vector<od::DetectedObject>& decodedResults,
                                                       cv::Mat imageMat)
{
    std::vector<std::tuple<int, int>> bboxes;
    for(const od::DetectedObject& object : decodedResults)
    {
        const od::BoundingBox& bbox = object.GetBoundingBox();

        if (bbox.GetX() + bbox.GetWidth() > imageMat.cols)
        {
            for (int y = bbox.GetY(); y < bbox.GetY() + bbox.GetHeight(); ++y)
            {
                bboxes.emplace_back(std::tuple<int, int>{bbox.GetX(), y});
            }

            for (int x = bbox.GetX(); x < imageMat.cols; ++x)
            {
                bboxes.emplace_back(std::tuple<int, int>{x, bbox.GetY() + bbox.GetHeight() - 1});
            }

            for (int y = bbox.GetY(); y < bbox.GetY() + bbox.GetHeight(); ++y)
            {
                bboxes.emplace_back(std::tuple<int, int>{imageMat.cols - 1, y});
            }
        }
        else if (bbox.GetY() + bbox.GetHeight() > imageMat.rows)
        {
            for (int y = bbox.GetY(); y < imageMat.rows; ++y)
            {
                bboxes.emplace_back(std::tuple<int, int>{bbox.GetX(), y});
            }

            for (int x = bbox.GetX(); x < bbox.GetX() + bbox.GetWidth(); ++x)
            {
                bboxes.emplace_back(std::tuple<int, int>{x, imageMat.rows - 1});
            }

            for (int y = bbox.GetY(); y < imageMat.rows; ++y)
            {
                bboxes.emplace_back(std::tuple<int, int>{bbox.GetX() + bbox.GetWidth() - 1, y});
            }
        }
        else
        {
            for (int y = bbox.GetY(); y < bbox.GetY() + bbox.GetHeight(); ++y)
            {
                bboxes.emplace_back(std::tuple<int, int>{bbox.GetX(), y});
            }

            for (int x = bbox.GetX(); x < bbox.GetX() + bbox.GetWidth(); ++x)
            {
                bboxes.emplace_back(std::tuple<int, int>{x, bbox.GetY() + bbox.GetHeight() - 1});
            }

            for (int y = bbox.GetY(); y < bbox.GetY() + bbox.GetHeight(); ++y)
            {
                bboxes.emplace_back(std::tuple<int, int>{bbox.GetX() + bbox.GetWidth() - 1, y});
            }
        }
    }
    return bboxes;
}

static std::string GetResourceFilePath(std::string filename)
{
    std::string testResources = TEST_RESOURCE_DIR;
    if (0 == testResources.size())
    {
        throw "Invalid test resources directory provided";
    }
    else
    {
        if(testResources.back() != '/')
        {
            return testResources + "/" + filename;
        }
        else
        {
            return testResources + filename;
        }
    }
}

TEST_CASE("Test Adding Inference output to frame")
{
    //todo: re-write test to use static detections

    std::string testResources = TEST_RESOURCE_DIR;
    REQUIRE(testResources != "");
    std::vector<std::tuple<std::string, common::BBoxColor>> labels;

    common::BBoxColor c
    {
        .colorCode = std::make_tuple (0, 0, 255)
    };

    auto bboxInfo = std::make_tuple ("person", c);
    od::BoundingBox bbox(10, 10, 50, 50);
    od::DetectedObject detection(0, "person", bbox, 0.75);

    labels.push_back(bboxInfo);

    od::DetectedObjects detections;
    cv::Mat frame = cv::imread(GetResourceFilePath("basketball1.png"), cv::IMREAD_COLOR);
    detections.push_back(detection);

    AddInferenceOutputToFrame(detections, frame, labels);

    std::vector<std::tuple<int, int>> bboxes = GetBoundingBoxPoints(detections, frame);

    // Check that every point is the expected color
    for(std::tuple<int, int> tuple : bboxes)
    {
        cv::Point p(std::get<0>(tuple), std::get<1>(tuple));
        CHECK(static_cast<int>(frame.at<cv::Vec3b>(p)[0]) == 0);
        CHECK(static_cast<int>(frame.at<cv::Vec3b>(p)[1]) == 0);
        CHECK(static_cast<int>(frame.at<cv::Vec3b>(p)[2]) == 255);
    }
}
