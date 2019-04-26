//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "InferenceTest.hpp"
#include "DeepSpeechV1Database.hpp"

#include <boost/assert.hpp>
#include <boost/log/trivial.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

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
        , m_FloatComparer(boost::math::fpc::percent_tolerance(1.0f))
        , m_ExpectedOutputs({testCaseData.m_ExpectedOutputData.m_InputSeq, testCaseData.m_ExpectedOutputData.m_StateH,
                             testCaseData.m_ExpectedOutputData.m_StateC})
    {}

    TestCaseResult ProcessResult(const InferenceTestOptions& options) override
    {
        const std::vector<float>& output1 = boost::get<std::vector<float>>(this->GetOutputs()[0]); // logits
        BOOST_ASSERT(output1.size() == k_OutputSize1);

        const std::vector<float>& output2 = boost::get<std::vector<float>>(this->GetOutputs()[1]); // new_state_c
        BOOST_ASSERT(output2.size() == k_OutputSize2);

        const std::vector<float>& output3 = boost::get<std::vector<float>>(this->GetOutputs()[2]); // new_state_h
        BOOST_ASSERT(output3.size() == k_OutputSize3);

        // Check each output to see whether it is the expected value
        for (unsigned int j = 0u; j < output1.size(); j++)
        {
            if(!m_FloatComparer(output1[j], m_ExpectedOutputs.m_InputSeq[j]))
            {
                BOOST_LOG_TRIVIAL(error) << "InputSeq for Lstm " << this->GetTestCaseId() <<
                                         " is incorrect at" << j;
                return TestCaseResult::Failed;
            }
        }

        for (unsigned int j = 0u; j < output2.size(); j++)
        {
            if(!m_FloatComparer(output2[j], m_ExpectedOutputs.m_StateH[j]))
            {
                BOOST_LOG_TRIVIAL(error) << "StateH for Lstm " << this->GetTestCaseId() <<
                                         " is incorrect";
                return TestCaseResult::Failed;
            }
        }

        for (unsigned int j = 0u; j < output3.size(); j++)
        {
            if(!m_FloatComparer(output3[j], m_ExpectedOutputs.m_StateC[j]))
            {
                BOOST_LOG_TRIVIAL(error) << "StateC for Lstm " << this->GetTestCaseId() <<
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

    boost::math::fpc::close_at_tolerance<float> m_FloatComparer;
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

    virtual void AddCommandLineOptions(boost::program_options::options_description& options) override
    {
        namespace po = boost::program_options;

        options.add_options()
                ("input-seq-dir,s", po::value<std::string>(&m_InputSeqDir)->required(),
                 "Path to directory containing test data for m_InputSeq");
        options.add_options()
                ("prev-state-h-dir,h", po::value<std::string>(&m_PrevStateHDir)->required(),
                 "Path to directory containing test data for m_PrevStateH");
        options.add_options()
                ("prev-state-c-dir,c", po::value<std::string>(&m_PrevStateCDir)->required(),
                 "Path to directory containing test data for m_PrevStateC");
        options.add_options()
                ("logits-dir,l", po::value<std::string>(&m_LogitsDir)->required(),
                 "Path to directory containing test data for m_Logits");
        options.add_options()
                ("new-state-h-dir,H", po::value<std::string>(&m_NewStateHDir)->required(),
                 "Path to directory containing test data for m_NewStateH");
        options.add_options()
                ("new-state-c-dir,C", po::value<std::string>(&m_NewStateCDir)->required(),
                 "Path to directory containing test data for m_NewStateC");


        Model::AddCommandLineOptions(options, m_ModelCommandLineOptions);
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
