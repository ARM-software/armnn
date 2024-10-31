//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "InferenceTest.hpp"
#include "YoloDatabase.hpp"

#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnnUtils/FloatingPointComparison.hpp>

#include <algorithm>
#include <array>
#include <utility>

constexpr size_t YoloOutputSize = 1470;

template <typename Model>
class YoloTestCase : public InferenceModelTestCase<Model>
{
public:
    YoloTestCase(Model& model,
        unsigned int testCaseId,
        YoloTestCaseData& testCaseData)
     : InferenceModelTestCase<Model>(model, testCaseId, { std::move(testCaseData.m_InputImage) }, { YoloOutputSize })
     , m_TopObjectDetections(std::move(testCaseData.m_TopObjectDetections))
    {
    }

    virtual TestCaseResult ProcessResult(const InferenceTestOptions& options) override
    {
        armnn::IgnoreUnused(options);

        const std::vector<float>& output = mapbox::util::get<std::vector<float>>(this->GetOutputs()[0]);
        ARMNN_ASSERT(output.size() == YoloOutputSize);

        constexpr unsigned int gridSize = 7;
        constexpr unsigned int numClasses = 20;
        constexpr unsigned int numScales = 2;

        const float* outputPtr =  output.data();

        // Range 0-980. Class probabilities. 7x7x20
        vector<vector<vector<float>>> classProbabilities(gridSize, vector<vector<float>>(gridSize,
                                                         vector<float>(numClasses)));
        for (unsigned int y = 0; y < gridSize; ++y)
        {
            for (unsigned int x = 0; x < gridSize; ++x)
            {
                for (unsigned int c = 0; c < numClasses; ++c)
                {
                    classProbabilities[y][x][c] = *outputPtr++;
                }
            }
        }

        // Range 980-1078. Scales. 7x7x2
        vector<vector<vector<float>>> scales(gridSize, vector<vector<float>>(gridSize, vector<float>(numScales)));
        for (unsigned int y = 0; y < gridSize; ++y)
        {
            for (unsigned int x = 0; x < gridSize; ++x)
            {
                for (unsigned int s = 0; s < numScales; ++s)
                {
                    scales[y][x][s] = *outputPtr++;
                }
            }
        }

        // Range 1078-1469. Bounding boxes. 7x7x2x4
        constexpr float imageWidthAsFloat = static_cast<float>(YoloImageWidth);
        constexpr float imageHeightAsFloat = static_cast<float>(YoloImageHeight);

        vector<vector<vector<vector<float>>>> boxes(gridSize, vector<vector<vector<float>>>
            (gridSize, vector<vector<float>>(numScales, vector<float>(4))));
        for (unsigned int y = 0; y < gridSize; ++y)
        {
            for (unsigned int x = 0; x < gridSize; ++x)
            {
                for (unsigned int s = 0; s < numScales; ++s)
                {
                    float bx = *outputPtr++;
                    float by = *outputPtr++;
                    float bw = *outputPtr++;
                    float bh = *outputPtr++;

                    boxes[y][x][s][0] = ((bx + static_cast<float>(x)) / 7.0f) * imageWidthAsFloat;
                    boxes[y][x][s][1] = ((by + static_cast<float>(y)) / 7.0f) * imageHeightAsFloat;
                    boxes[y][x][s][2] = bw * bw * static_cast<float>(imageWidthAsFloat);
                    boxes[y][x][s][3] = bh * bh * static_cast<float>(imageHeightAsFloat);
                }
            }
        }
        ARMNN_ASSERT(output.data() + YoloOutputSize == outputPtr);

        std::vector<YoloDetectedObject> detectedObjects;
        detectedObjects.reserve(gridSize * gridSize * numScales * numClasses);

        for (unsigned int y = 0; y < gridSize; ++y)
        {
            for (unsigned int x = 0; x < gridSize; ++x)
            {
                for (unsigned int s = 0; s < numScales; ++s)
                {
                    for (unsigned int c = 0; c < numClasses; ++c)
                    {
                        // Resolved confidence: class probabilities * scales.
                        const float confidence = classProbabilities[y][x][c] * scales[y][x][s];

                        // Resolves bounding box and stores.
                        YoloBoundingBox box;
                        box.m_X = boxes[y][x][s][0];
                        box.m_Y = boxes[y][x][s][1];
                        box.m_W = boxes[y][x][s][2];
                        box.m_H = boxes[y][x][s][3];

                        detectedObjects.emplace_back(c, box, confidence);
                    }
                }
            }
        }

        // Sorts detected objects by confidence.
        std::sort(detectedObjects.begin(), detectedObjects.end(),
            [](const YoloDetectedObject& a, const YoloDetectedObject& b)
            {
                // Sorts by largest confidence first, then by class.
                return a.m_Confidence > b.m_Confidence
                    || (a.m_Confidence == b.m_Confidence && a.m_Class > b.m_Class);
            });

        // Checks the top N detections.
        auto outputIt  = detectedObjects.begin();
        auto outputEnd = detectedObjects.end();

        for (const YoloDetectedObject& expectedDetection : m_TopObjectDetections)
        {
            if (outputIt == outputEnd)
            {
                // Somehow expected more things to check than detections found by the model.
                return TestCaseResult::Abort;
            }

            const YoloDetectedObject& detectedObject = *outputIt;
            if (detectedObject.m_Class != expectedDetection.m_Class)
            {
                ARMNN_LOG(error) << "Prediction for test case " << this->GetTestCaseId() <<
                    " is incorrect: Expected (" << expectedDetection.m_Class << ")" <<
                    " but predicted (" << detectedObject.m_Class << ")";
                return TestCaseResult::Failed;
            }

            if (!armnnUtils::within_percentage_tolerance(detectedObject.m_Box.m_X, expectedDetection.m_Box.m_X) ||
                !armnnUtils::within_percentage_tolerance(detectedObject.m_Box.m_Y, expectedDetection.m_Box.m_Y) ||
                !armnnUtils::within_percentage_tolerance(detectedObject.m_Box.m_W, expectedDetection.m_Box.m_W) ||
                !armnnUtils::within_percentage_tolerance(detectedObject.m_Box.m_H, expectedDetection.m_Box.m_H) ||
                !armnnUtils::within_percentage_tolerance(detectedObject.m_Confidence, expectedDetection.m_Confidence))
            {
                ARMNN_LOG(error) << "Detected bounding box for test case " << this->GetTestCaseId() <<
                    " is incorrect";
                return TestCaseResult::Failed;
            }

            ++outputIt;
        }

        return TestCaseResult::Ok;
    }

private:
    std::vector<YoloDetectedObject> m_TopObjectDetections;
};

template <typename Model>
class YoloTestCaseProvider : public IInferenceTestCaseProvider
{
public:
    template <typename TConstructModelCallable>
    explicit YoloTestCaseProvider(TConstructModelCallable constructModel)
        : m_ConstructModel(constructModel)
    {
    }

    virtual void AddCommandLineOptions(cxxopts::Options& options, std::vector<std::string>& required) override
    {
        options
            .allow_unrecognised_options()
            .add_options()
                ("d,data-dir", "Path to directory containing test data", cxxopts::value<std::string>(m_DataDir));

        Model::AddCommandLineOptions(options, m_ModelCommandLineOptions, required);
    }

    virtual bool ProcessCommandLineOptions(const InferenceTestOptions& commonOptions) override
    {
        if (!ValidateDirectory(m_DataDir))
        {
            return false;
        }

        m_Model = m_ConstructModel(commonOptions, m_ModelCommandLineOptions);
        if (!m_Model)
        {
            return false;
        }

        m_Database = std::make_unique<YoloDatabase>(m_DataDir.c_str());
        if (!m_Database)
        {
            return false;
        }

        return true;
    }

    virtual std::unique_ptr<IInferenceTestCase> GetTestCase(unsigned int testCaseId) override
    {
        std::unique_ptr<YoloTestCaseData> testCaseData = m_Database->GetTestCaseData(testCaseId);
        if (!testCaseData)
        {
            return nullptr;
        }

        return std::make_unique<YoloTestCase<Model>>(*m_Model, testCaseId, *testCaseData);
    }

private:
    typename Model::CommandLineOptions m_ModelCommandLineOptions;
    std::function<std::unique_ptr<Model>(const InferenceTestOptions&,
                                         typename Model::CommandLineOptions)> m_ConstructModel;
    std::unique_ptr<Model> m_Model;

    std::string m_DataDir;
    std::unique_ptr<YoloDatabase> m_Database;
};
