//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "InferenceTest.hpp"

#include <armnn/Utils.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnnUtils/TContainer.hpp>

#include "CxxoptsUtils.hpp"

#include <cxxopts/cxxopts.hpp>
#include <fmt/format.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <array>
#include <chrono>

using namespace std;
using namespace std::chrono;
using namespace armnn::test;

namespace armnn
{
namespace test
{

template <typename TTestCaseDatabase, typename TModel>
ClassifierTestCase<TTestCaseDatabase, TModel>::ClassifierTestCase(
    int& numInferencesRef,
    int& numCorrectInferencesRef,
    const std::vector<unsigned int>& validationPredictions,
    std::vector<unsigned int>* validationPredictionsOut,
    TModel& model,
    unsigned int testCaseId,
    unsigned int label,
    std::vector<typename TModel::DataType> modelInput)
    : InferenceModelTestCase<TModel>(
            model, testCaseId, std::vector<armnnUtils::TContainer>{ modelInput }, { model.GetOutputSize() })
    , m_Label(label)
    , m_QuantizationParams(model.GetQuantizationParams())
    , m_NumInferencesRef(numInferencesRef)
    , m_NumCorrectInferencesRef(numCorrectInferencesRef)
    , m_ValidationPredictions(validationPredictions)
    , m_ValidationPredictionsOut(validationPredictionsOut)
{
}

struct ClassifierResultProcessor
{
    using ResultMap = std::map<float,int>;

    ClassifierResultProcessor(float scale, int offset)
        : m_Scale(scale)
        , m_Offset(offset)
    {}

    void operator()(const std::vector<float>& values)
    {
        SortPredictions(values, [](float value)
                                {
                                    return value;
                                });
    }

    void operator()(const std::vector<int8_t>& values)
    {
        SortPredictions(values, [](int8_t value)
        {
            return value;
        });
    }

    void operator()(const std::vector<uint8_t>& values)
    {
        auto& scale = m_Scale;
        auto& offset = m_Offset;
        SortPredictions(values, [&scale, &offset](uint8_t value)
                                {
                                    return armnn::Dequantize(value, scale, offset);
                                });
    }

    void operator()(const std::vector<int>& values)
    {
        IgnoreUnused(values);
        ARMNN_ASSERT_MSG(false, "Non-float predictions output not supported.");
    }

    ResultMap& GetResultMap() { return m_ResultMap; }

private:
    template<typename Container, typename Delegate>
    void SortPredictions(const Container& c, Delegate delegate)
    {
        int index = 0;
        for (const auto& value : c)
        {
            int classification = index++;
            // Take the first class with each probability
            // This avoids strange results when looping over batched results produced
            // with identical test data.
            ResultMap::iterator lb = m_ResultMap.lower_bound(value);

            if (lb == m_ResultMap.end() || !m_ResultMap.key_comp()(value, lb->first))
            {
                // If the key is not already in the map, insert it.
                m_ResultMap.insert(lb, ResultMap::value_type(delegate(value), classification));
            }
        }
    }

    ResultMap m_ResultMap;

    float m_Scale=0.0f;
    int m_Offset=0;
};

template <typename TTestCaseDatabase, typename TModel>
TestCaseResult ClassifierTestCase<TTestCaseDatabase, TModel>::ProcessResult(const InferenceTestOptions& params)
{
    auto& output = this->GetOutputs()[0];
    const auto testCaseId = this->GetTestCaseId();

    ClassifierResultProcessor resultProcessor(m_QuantizationParams.first, m_QuantizationParams.second);
    mapbox::util::apply_visitor(resultProcessor, output);

    ARMNN_LOG(info) << "= Prediction values for test #" << testCaseId;
    auto it = resultProcessor.GetResultMap().rbegin();
    for (int i=0; i<5 && it != resultProcessor.GetResultMap().rend(); ++i)
    {
        ARMNN_LOG(info) << "Top(" << (i+1) << ") prediction is " << it->second <<
          " with value: " << (it->first);
        ++it;
    }

    unsigned int prediction = 0;
    mapbox::util::apply_visitor([&](auto&& value)
                         {
                             prediction = armnn::numeric_cast<unsigned int>(
                                     std::distance(value.begin(), std::max_element(value.begin(), value.end())));
                         },
                         output);

    // If we're just running the defaultTestCaseIds, each one must be classified correctly.
    if (params.m_IterationCount == 0 && prediction != m_Label)
    {
        ARMNN_LOG(error) << "Prediction for test case " << testCaseId << " (" << prediction << ")" <<
            " is incorrect (should be " << m_Label << ")";
        return TestCaseResult::Failed;
    }

    // If a validation file was provided as input, it checks that the prediction matches.
    if (!m_ValidationPredictions.empty() && prediction != m_ValidationPredictions[testCaseId])
    {
        ARMNN_LOG(error) << "Prediction for test case " << testCaseId << " (" << prediction << ")" <<
            " doesn't match the prediction in the validation file (" << m_ValidationPredictions[testCaseId] << ")";
        return TestCaseResult::Failed;
    }

    // If a validation file was requested as output, it stores the predictions.
    if (m_ValidationPredictionsOut)
    {
        m_ValidationPredictionsOut->push_back(prediction);
    }

    // Updates accuracy stats.
    m_NumInferencesRef++;
    if (prediction == m_Label)
    {
        m_NumCorrectInferencesRef++;
    }

    return TestCaseResult::Ok;
}

template <typename TDatabase, typename InferenceModel>
template <typename TConstructDatabaseCallable, typename TConstructModelCallable>
ClassifierTestCaseProvider<TDatabase, InferenceModel>::ClassifierTestCaseProvider(
    TConstructDatabaseCallable constructDatabase, TConstructModelCallable constructModel)
    : m_ConstructModel(constructModel)
    , m_ConstructDatabase(constructDatabase)
    , m_NumInferences(0)
    , m_NumCorrectInferences(0)
{
}

template <typename TDatabase, typename InferenceModel>
void ClassifierTestCaseProvider<TDatabase, InferenceModel>::AddCommandLineOptions(
    cxxopts::Options& options, std::vector<std::string>& required)
{
    options
        .allow_unrecognised_options()
        .add_options()
            ("validation-file-in",
             "Reads expected predictions from the given file and confirms they match the actual predictions.",
             cxxopts::value<std::string>(m_ValidationFileIn)->default_value(""))
            ("validation-file-out", "Predictions are saved to the given file for later use via --validation-file-in.",
             cxxopts::value<std::string>(m_ValidationFileOut)->default_value(""))
            ("d,data-dir", "Path to directory containing test data", cxxopts::value<std::string>(m_DataDir));

    required.emplace_back("data-dir"); //add to required arguments to check

    InferenceModel::AddCommandLineOptions(options, m_ModelCommandLineOptions, required);
}

template <typename TDatabase, typename InferenceModel>
bool ClassifierTestCaseProvider<TDatabase, InferenceModel>::ProcessCommandLineOptions(
        const InferenceTestOptions& commonOptions)
{
    if (!ValidateDirectory(m_DataDir))
    {
        return false;
    }

    ReadPredictions();

    m_Model = m_ConstructModel(commonOptions, m_ModelCommandLineOptions);
    if (!m_Model)
    {
        return false;
    }

    m_Database = std::make_unique<TDatabase>(m_ConstructDatabase(m_DataDir.c_str(), *m_Model));
    if (!m_Database)
    {
        return false;
    }

    return true;
}

template <typename TDatabase, typename InferenceModel>
std::unique_ptr<IInferenceTestCase>
ClassifierTestCaseProvider<TDatabase, InferenceModel>::GetTestCase(unsigned int testCaseId)
{
    std::unique_ptr<typename TDatabase::TTestCaseData> testCaseData = m_Database->GetTestCaseData(testCaseId);
    if (testCaseData == nullptr)
    {
        return nullptr;
    }

    return std::make_unique<ClassifierTestCase<TDatabase, InferenceModel>>(
        m_NumInferences,
        m_NumCorrectInferences,
        m_ValidationPredictions,
        m_ValidationFileOut.empty() ? nullptr : &m_ValidationPredictionsOut,
        *m_Model,
        testCaseId,
        testCaseData->m_Label,
        std::move(testCaseData->m_InputImage));
}

template <typename TDatabase, typename InferenceModel>
bool ClassifierTestCaseProvider<TDatabase, InferenceModel>::OnInferenceTestFinished()
{
    const double accuracy = armnn::numeric_cast<double>(m_NumCorrectInferences) /
        armnn::numeric_cast<double>(m_NumInferences);
    ARMNN_LOG(info) << std::fixed << std::setprecision(3) << "Overall accuracy: " << accuracy;

    // If a validation file was requested as output, the predictions are saved to it.
    if (!m_ValidationFileOut.empty())
    {
        std::ofstream validationFileOut(m_ValidationFileOut.c_str(), std::ios_base::trunc | std::ios_base::out);
        if (validationFileOut.good())
        {
            for (const unsigned int prediction : m_ValidationPredictionsOut)
            {
                validationFileOut << prediction << std::endl;
            }
        }
        else
        {
            ARMNN_LOG(error) << "Failed to open output validation file: " << m_ValidationFileOut;
            return false;
        }
    }

    return true;
}

template <typename TDatabase, typename InferenceModel>
void ClassifierTestCaseProvider<TDatabase, InferenceModel>::ReadPredictions()
{
    // Reads the expected predictions from the input validation file (if provided).
    if (!m_ValidationFileIn.empty())
    {
        std::ifstream validationFileIn(m_ValidationFileIn.c_str(), std::ios_base::in);
        if (validationFileIn.good())
        {
            while (!validationFileIn.eof())
            {
                unsigned int i;
                validationFileIn >> i;
                m_ValidationPredictions.emplace_back(i);
            }
        }
        else
        {
            throw armnn::Exception(fmt::format("Failed to open input validation file: {}"
                , m_ValidationFileIn));
        }
    }
}

template<typename TConstructTestCaseProvider>
int InferenceTestMain(int argc,
    char* argv[],
    const std::vector<unsigned int>& defaultTestCaseIds,
    TConstructTestCaseProvider constructTestCaseProvider)
{
    // Configures logging for both the ARMNN library and this test program.
#ifdef NDEBUG
    armnn::LogSeverity level = armnn::LogSeverity::Info;
#else
    armnn::LogSeverity level = armnn::LogSeverity::Debug;
#endif
    armnn::ConfigureLogging(true, true, level);

    try
    {
        std::unique_ptr<IInferenceTestCaseProvider> testCaseProvider = constructTestCaseProvider();
        if (!testCaseProvider)
        {
            return 1;
        }

        InferenceTestOptions inferenceTestOptions;
        if (!ParseCommandLine(argc, argv, *testCaseProvider, inferenceTestOptions))
        {
            return 1;
        }

        const bool success = InferenceTest(inferenceTestOptions, defaultTestCaseIds, *testCaseProvider);
        return success ? 0 : 1;
    }
    catch (armnn::Exception const& e)
    {
        ARMNN_LOG(fatal) << "Armnn Error: " << e.what();
        return 1;
    }
}

//
// This function allows us to create a classifier inference test based on:
//  - a model file name
//  - which can be a binary or a text file for protobuf formats
//  - an input tensor name
//  - an output tensor name
//  - a set of test case ids
//  - a callback method which creates an object that can return images
//    called 'Database' in these tests
//  - and an input tensor shape
//
template<typename TDatabase,
         typename TParser,
         typename TConstructDatabaseCallable>
int ClassifierInferenceTestMain(int argc,
                                char* argv[],
                                const char* modelFilename,
                                bool isModelBinary,
                                const char* inputBindingName,
                                const char* outputBindingName,
                                const std::vector<unsigned int>& defaultTestCaseIds,
                                TConstructDatabaseCallable constructDatabase,
                                const armnn::TensorShape* inputTensorShape)

{
    ARMNN_ASSERT(modelFilename);
    ARMNN_ASSERT(inputBindingName);
    ARMNN_ASSERT(outputBindingName);

    return InferenceTestMain(argc, argv, defaultTestCaseIds,
        [=]
        ()
        {
            using InferenceModel = InferenceModel<TParser, typename TDatabase::DataType>;
            using TestCaseProvider = ClassifierTestCaseProvider<TDatabase, InferenceModel>;

            return make_unique<TestCaseProvider>(constructDatabase,
                [&]
                (const InferenceTestOptions &commonOptions,
                 typename InferenceModel::CommandLineOptions modelOptions)
                {
                    if (!ValidateDirectory(modelOptions.m_ModelDir))
                    {
                        return std::unique_ptr<InferenceModel>();
                    }

                    typename InferenceModel::Params modelParams;
                    modelParams.m_ModelPath = modelOptions.m_ModelDir + modelFilename;
                    modelParams.m_InputBindings  = { inputBindingName };
                    modelParams.m_OutputBindings = { outputBindingName };

                    if (inputTensorShape)
                    {
                        modelParams.m_InputShapes.push_back(*inputTensorShape);
                    }

                    modelParams.m_IsModelBinary = isModelBinary;
                    modelParams.m_ComputeDevices = modelOptions.GetComputeDevicesAsBackendIds();
                    modelParams.m_VisualizePostOptimizationModel = modelOptions.m_VisualizePostOptimizationModel;
                    modelParams.m_EnableFp16TurboMode = modelOptions.m_EnableFp16TurboMode;

                    return std::make_unique<InferenceModel>(modelParams,
                                                            commonOptions.m_EnableProfiling,
                                                            commonOptions.m_DynamicBackendsPath);
            });
        });
}

} // namespace test
} // namespace armnn
