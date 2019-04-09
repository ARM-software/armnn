//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "InferenceTest.hpp"
#include "MobileNetSsdDatabase.hpp"

#include <boost/assert.hpp>
#include <boost/log/trivial.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

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
        , m_FloatComparer(boost::math::fpc::percent_tolerance(1.0f))
        , m_DetectedObjects(testCaseData.m_ExpectedDetectedObject)
    {}

    TestCaseResult ProcessResult(const InferenceTestOptions& options) override
    {
        const std::vector<float>& output1 = boost::get<std::vector<float>>(this->GetOutputs()[0]); // bounding boxes
        BOOST_ASSERT(output1.size() == k_OutputSize1);

        const std::vector<float>& output2 = boost::get<std::vector<float>>(this->GetOutputs()[1]); // classes
        BOOST_ASSERT(output2.size() == k_OutputSize2);

        const std::vector<float>& output3 = boost::get<std::vector<float>>(this->GetOutputs()[2]); // scores
        BOOST_ASSERT(output3.size() == k_OutputSize3);

        const std::vector<float>& output4 = boost::get<std::vector<float>>(this->GetOutputs()[3]); // valid detections
        BOOST_ASSERT(output4.size() == k_OutputSize4);

        const size_t numDetections = boost::numeric_cast<size_t>(output4[0]);

        // Check if number of valid detections matches expectations
        const size_t expectedNumDetections = m_DetectedObjects.size();
        if (numDetections != expectedNumDetections)
        {
            BOOST_LOG_TRIVIAL(error) << "Number of detections is incorrect: Expected (" <<
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
                BOOST_LOG_TRIVIAL(error) << "No more detected objects found! Index out of bounds: " << i;
                return TestCaseResult::Abort;
            }

            const DetectedObject& detectedObject = *it;
            const DetectedObject& expectedObject = m_DetectedObjects[i];

            if (detectedObject.m_Class != expectedObject.m_Class)
            {
                BOOST_LOG_TRIVIAL(error) << "Prediction for test case " << this->GetTestCaseId() <<
                    " is incorrect: Expected (" << expectedObject.m_Class << ")" <<
                    " but predicted (" << detectedObject.m_Class << ")";
                return TestCaseResult::Failed;
            }

            if(!m_FloatComparer(detectedObject.m_Confidence, expectedObject.m_Confidence))
            {
                BOOST_LOG_TRIVIAL(error) << "Confidence of prediction for test case " << this->GetTestCaseId() <<
                    " is incorrect: Expected (" << expectedObject.m_Confidence << ")  +- 1.0 pc" <<
                    " but predicted (" << detectedObject.m_Confidence << ")";
                return TestCaseResult::Failed;
            }

            if (!m_FloatComparer(detectedObject.m_BoundingBox.m_XMin, expectedObject.m_BoundingBox.m_XMin) ||
                !m_FloatComparer(detectedObject.m_BoundingBox.m_YMin, expectedObject.m_BoundingBox.m_YMin) ||
                !m_FloatComparer(detectedObject.m_BoundingBox.m_XMax, expectedObject.m_BoundingBox.m_XMax) ||
                !m_FloatComparer(detectedObject.m_BoundingBox.m_YMax, expectedObject.m_BoundingBox.m_YMax))
            {
                BOOST_LOG_TRIVIAL(error) << "Detected bounding box for test case " << this->GetTestCaseId() <<
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

    boost::math::fpc::close_at_tolerance<float> m_FloatComparer;
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

    virtual void AddCommandLineOptions(boost::program_options::options_description& options) override
    {
        namespace po = boost::program_options;

        options.add_options()
            ("data-dir,d", po::value<std::string>(&m_DataDir)->required(),
             "Path to directory containing test data");

        Model::AddCommandLineOptions(options, m_ModelCommandLineOptions);
    }

    virtual bool ProcessCommandLineOptions(const InferenceTestOptions &commonOptions) override
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