//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "InferenceTest.hpp"
#include "DeepSpeechV1Database.hpp"

#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnnUtils/FloatingPointComparison.hpp>

#include <vector>

namespace
{

template<typename Model>
class DeepSpeechV1TestCase : public InferenceModelTestCase<Model>
{
public:
    DeepSpeechV1TestCase(Model& model,
                         unsigned int testCaseId,
                         const DeepSpeechV1TestCaseData& testCaseData)
        : InferenceModelTestCase<Model>(model,
                                        testCaseId,
                                        { testCaseData.m_InputData.m_InputSeq,
                                          testCaseData.m_InputData.m_StateH,
                                          testCaseData.m_InputData.m_StateC},
                                        { k_OutputSize1, k_OutputSize2, k_OutputSize3 })
        , m_ExpectedOutputs({testCaseData.m_ExpectedOutputData.m_InputSeq, testCaseData.m_ExpectedOutputData.m_StateH,
                             testCaseData.m_ExpectedOutputData.m_StateC})
    {}

    TestCaseResult ProcessResult(const InferenceTestOptions& options) override
    {
        armnn::IgnoreUnused(options);
        const std::vector<float>& output1 = mapbox::util::get<std::vector<float>>(this->GetOutputs()[0]); // logits
        ARMNN_ASSERT(output1.size() == k_OutputSize1);

        const std::vector<float>& output2 = mapbox::util::get<std::vector<float>>(this->GetOutputs()[1]); // new_state_c
        ARMNN_ASSERT(output2.size() == k_OutputSize2);

        const std::vector<float>& output3 = mapbox::util::get<std::vector<float>>(this->GetOutputs()[2]); // new_state_h
        ARMNN_ASSERT(output3.size() == k_OutputSize3);

        // Check each output to see whether it is the expected value
        for (unsigned int j = 0u; j < output1.size(); j++)
        {
            if(!armnnUtils::within_percentage_tolerance(output1[j], m_ExpectedOutputs.m_InputSeq[j]))
            {
                ARMNN_LOG(error) << "InputSeq for Lstm " << this->GetTestCaseId() <<
                                         " is incorrect at" << j;
                return TestCaseResult::Failed;
            }
        }

        for (unsigned int j = 0u; j < output2.size(); j++)
        {
            if(!armnnUtils::within_percentage_tolerance(output2[j], m_ExpectedOutputs.m_StateH[j]))
            {
                ARMNN_LOG(error) << "StateH for Lstm " << this->GetTestCaseId() <<
                                         " is incorrect";
                return TestCaseResult::Failed;
            }
        }

        for (unsigned int j = 0u; j < output3.size(); j++)
        {
            if(!armnnUtils::within_percentage_tolerance(output3[j], m_ExpectedOutputs.m_StateC[j]))
            {
                ARMNN_LOG(error) << "StateC for Lstm " << this->GetTestCaseId() <<
                                         " is incorrect";
                return TestCaseResult::Failed;
            }
        }
        return TestCaseResult::Ok;
    }

private:

    static constexpr unsigned int k_OutputSize1 = 464u;
    static constexpr unsigned int k_OutputSize2 = 2048u;
    static constexpr unsigned int k_OutputSize3 = 2048u;

    LstmInput m_ExpectedOutputs;
};

template <typename Model>
class DeepSpeechV1TestCaseProvider : public IInferenceTestCaseProvider
{
public:
    template <typename TConstructModelCallable>
    explicit DeepSpeechV1TestCaseProvider(TConstructModelCallable constructModel)
        : m_ConstructModel(constructModel)
    {}

    virtual void AddCommandLineOptions(cxxopts::Options& options, std::vector<std::string>& required) override
    {
        options
            .allow_unrecognised_options()
            .add_options()
                ("s,input-seq-dir", "Path to directory containing test data for m_InputSeq",
                 cxxopts::value<std::string>(m_InputSeqDir))
                ("h,prev-state-h-dir", "Path to directory containing test data for m_PrevStateH",
                 cxxopts::value<std::string>(m_PrevStateHDir))
                ("c,prev-state-c-dir", "Path to directory containing test data for m_PrevStateC",
                 cxxopts::value<std::string>(m_PrevStateCDir))
                ("l,logits-dir", "Path to directory containing test data for m_Logits",
                 cxxopts::value<std::string>(m_LogitsDir))
                ("H,new-state-h-dir", "Path to directory containing test data for m_NewStateH",
                 cxxopts::value<std::string>(m_NewStateHDir))
                ("C,new-state-c-dir", "Path to directory containing test data for m_NewStateC",
                 cxxopts::value<std::string>(m_NewStateCDir));

        required.insert(required.end(), {"input-seq-dir", "prev-state-h-dir", "prev-state-c-dir", "logits-dir",
                                         "new-state-h-dir", "new-state-c-dir"});

        Model::AddCommandLineOptions(options, m_ModelCommandLineOptions, required);
    }

    virtual bool ProcessCommandLineOptions(const InferenceTestOptions &commonOptions) override
    {
        if (!ValidateDirectory(m_InputSeqDir))
        {
            return false;
        }

        if (!ValidateDirectory(m_PrevStateCDir))
        {
            return false;
        }

        if (!ValidateDirectory(m_PrevStateHDir))
        {
            return false;
        }

        if (!ValidateDirectory(m_LogitsDir))
        {
            return false;
        }

        if (!ValidateDirectory(m_NewStateCDir))
        {
            return false;
        }

        if (!ValidateDirectory(m_NewStateHDir))
        {
            return false;
        }

        m_Model = m_ConstructModel(commonOptions, m_ModelCommandLineOptions);
        if (!m_Model)
        {
            return false;
        }
        m_Database = std::make_unique<DeepSpeechV1Database>(m_InputSeqDir.c_str(), m_PrevStateHDir.c_str(),
                                                            m_PrevStateCDir.c_str(), m_LogitsDir.c_str(),
                                                            m_NewStateHDir.c_str(), m_NewStateCDir.c_str());
        if (!m_Database)
        {
            return false;
        }

        return true;
    }

    std::unique_ptr<IInferenceTestCase> GetTestCase(unsigned int testCaseId) override
    {
        std::unique_ptr<DeepSpeechV1TestCaseData> testCaseData = m_Database->GetTestCaseData(testCaseId);
        if (!testCaseData)
        {
            return nullptr;
        }

        return std::make_unique<DeepSpeechV1TestCase<Model>>(*m_Model, testCaseId, *testCaseData);
    }

private:
    typename Model::CommandLineOptions m_ModelCommandLineOptions;
    std::function<std::unique_ptr<Model>(const InferenceTestOptions&,
                                         typename Model::CommandLineOptions)> m_ConstructModel;
    std::unique_ptr<Model> m_Model;

    std::string m_InputSeqDir;
    std::string m_PrevStateCDir;
    std::string m_PrevStateHDir;
    std::string m_LogitsDir;
    std::string m_NewStateCDir;
    std::string m_NewStateHDir;

    std::unique_ptr<DeepSpeechV1Database> m_Database;
};

} // anonymous namespace
