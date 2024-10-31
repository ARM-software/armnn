//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "InferenceTest.hpp"
#include "MobileNetSsdDatabase.hpp"

#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnnUtils/FloatingPointComparison.hpp>

#include <vector>

namespace
{

template<typename Model>
class MobileNetSsdTestCase : public InferenceModelTestCase<Model>
{
public:
    MobileNetSsdTestCase(Model& model,
                         unsigned int testCaseId,
                         const MobileNetSsdTestCaseData& testCaseData)
        : InferenceModelTestCase<Model>(model,
                                        testCaseId,
                                        { std::move(testCaseData.m_InputData) },
                                        { k_OutputSize1, k_OutputSize2, k_OutputSize3, k_OutputSize4 })
        , m_DetectedObjects(testCaseData.m_ExpectedDetectedObject)
    {}

    TestCaseResult ProcessResult(const InferenceTestOptions& options) override
    {
        armnn::IgnoreUnused(options);

        // bounding boxes
        const std::vector<float>& output1 = mapbox::util::get<std::vector<float>>(this->GetOutputs()[0]);
        ARMNN_ASSERT(output1.size() == k_OutputSize1);

        // classes
        const std::vector<float>& output2 = mapbox::util::get<std::vector<float>>(this->GetOutputs()[1]);
        ARMNN_ASSERT(output2.size() == k_OutputSize2);

        // scores
        const std::vector<float>& output3 = mapbox::util::get<std::vector<float>>(this->GetOutputs()[2]);
        ARMNN_ASSERT(output3.size() == k_OutputSize3);

        // valid detections
        const std::vector<float>& output4 = mapbox::util::get<std::vector<float>>(this->GetOutputs()[3]);
        ARMNN_ASSERT(output4.size() == k_OutputSize4);

        const size_t numDetections = armnn::numeric_cast<size_t>(output4[0]);

        // Check if number of valid detections matches expectations
        const size_t expectedNumDetections = m_DetectedObjects.size();
        if (numDetections != expectedNumDetections)
        {
            ARMNN_LOG(error) << "Number of detections is incorrect: Expected (" <<
                expectedNumDetections << ")" << " but got (" << numDetections << ")";
            return TestCaseResult::Failed;
        }

        // Extract detected objects from output data
        std::vector<DetectedObject> detectedObjects;
        const float* outputData = output1.data();
        for (unsigned int i = 0u; i < numDetections; i++)
        {
            // NOTE: Order of coordinates in output data is yMin, xMin, yMax, xMax
            float yMin = *outputData++;
            float xMin = *outputData++;
            float yMax = *outputData++;
            float xMax = *outputData++;

            DetectedObject detectedObject(
                output2.at(i),
                BoundingBox(xMin, yMin, xMax, yMax),
                output3.at(i));

            detectedObjects.push_back(detectedObject);
        }

        std::sort(detectedObjects.begin(), detectedObjects.end());
        std::sort(m_DetectedObjects.begin(), m_DetectedObjects.end());

        // Compare detected objects with expected results
        std::vector<DetectedObject>::const_iterator it = detectedObjects.begin();
        for (unsigned int i = 0; i < numDetections; i++)
        {
            if (it == detectedObjects.end())
            {
                ARMNN_LOG(error) << "No more detected objects found! Index out of bounds: " << i;
                return TestCaseResult::Abort;
            }

            const DetectedObject& detectedObject = *it;
            const DetectedObject& expectedObject = m_DetectedObjects[i];

            if (detectedObject.m_Class != expectedObject.m_Class)
            {
                ARMNN_LOG(error) << "Prediction for test case " << this->GetTestCaseId() <<
                    " is incorrect: Expected (" << expectedObject.m_Class << ")" <<
                    " but predicted (" << detectedObject.m_Class << ")";
                return TestCaseResult::Failed;
            }

            if(!armnnUtils::within_percentage_tolerance(detectedObject.m_Confidence, expectedObject.m_Confidence))
            {
                ARMNN_LOG(error) << "Confidence of prediction for test case " << this->GetTestCaseId() <<
                    " is incorrect: Expected (" << expectedObject.m_Confidence << ")  +- 1.0 pc" <<
                    " but predicted (" << detectedObject.m_Confidence << ")";
                return TestCaseResult::Failed;
            }

            if (!armnnUtils::within_percentage_tolerance(detectedObject.m_BoundingBox.m_XMin,
                                                         expectedObject.m_BoundingBox.m_XMin) ||
                !armnnUtils::within_percentage_tolerance(detectedObject.m_BoundingBox.m_YMin,
                                                         expectedObject.m_BoundingBox.m_YMin) ||
                !armnnUtils::within_percentage_tolerance(detectedObject.m_BoundingBox.m_XMax,
                                                         expectedObject.m_BoundingBox.m_XMax) ||
                !armnnUtils::within_percentage_tolerance(detectedObject.m_BoundingBox.m_YMax,
                                                         expectedObject.m_BoundingBox.m_YMax))
            {
                ARMNN_LOG(error) << "Detected bounding box for test case " << this->GetTestCaseId() <<
                    " is incorrect";
                return TestCaseResult::Failed;
            }

            ++it;
        }

        return TestCaseResult::Ok;
    }

private:
    static constexpr unsigned int k_Shape       = 10u;

    static constexpr unsigned int k_OutputSize1 = k_Shape * 4u;
    static constexpr unsigned int k_OutputSize2 = k_Shape;
    static constexpr unsigned int k_OutputSize3 = k_Shape;
    static constexpr unsigned int k_OutputSize4 = 1u;

    std::vector<DetectedObject>                 m_DetectedObjects;
};

template <typename Model>
class MobileNetSsdTestCaseProvider : public IInferenceTestCaseProvider
{
public:
    template <typename TConstructModelCallable>
    explicit MobileNetSsdTestCaseProvider(TConstructModelCallable constructModel)
        : m_ConstructModel(constructModel)
    {}

    virtual void AddCommandLineOptions(cxxopts::Options& options, std::vector<std::string>& required) override
    {
        options
            .allow_unrecognised_options()
            .add_options()
                ("d,data-dir", "Path to directory containing test data", cxxopts::value<std::string>(m_DataDir));

        required.emplace_back("data-dir");

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
        std::pair<float, int32_t> qParams = m_Model->GetInputQuantizationParams();
        m_Database = std::make_unique<MobileNetSsdDatabase>(m_DataDir.c_str(), qParams.first, qParams.second);
        if (!m_Database)
        {
            return false;
        }

        return true;
    }

    std::unique_ptr<IInferenceTestCase> GetTestCase(unsigned int testCaseId) override
    {
        std::unique_ptr<MobileNetSsdTestCaseData> testCaseData = m_Database->GetTestCaseData(testCaseId);
        if (!testCaseData)
        {
            return nullptr;
        }

        return std::make_unique<MobileNetSsdTestCase<Model>>(*m_Model, testCaseId, *testCaseData);
    }

private:
    typename Model::CommandLineOptions m_ModelCommandLineOptions;
    std::function<std::unique_ptr<Model>(const InferenceTestOptions &,
                                         typename Model::CommandLineOptions)> m_ConstructModel;
    std::unique_ptr<Model> m_Model;

    std::string m_DataDir;
    std::unique_ptr<MobileNetSsdDatabase> m_Database;
};

} // anonymous namespace