//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct PassThruFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    PassThruFixture()
    {
        m_Prototext = "node {\n"
            "  name: \"Placeholder\"\n"
            "  op: \"Placeholder\"\n"
            "  attr {\n"
            "    key: \"dtype\"\n"
            "    value {\n"
            "      type: DT_FLOAT\n"
            "    }\n"
            "  }\n"
            "  attr {\n"
            "    key: \"shape\"\n"
            "    value {\n"
            "      shape {\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "}\n";
        SetupSingleInputSingleOutput({ 1, 7 }, "Placeholder", "Placeholder");
    }
};

BOOST_FIXTURE_TEST_CASE(ValidateOutput, PassThruFixture)
{
    BOOST_TEST(m_Parser->GetNetworkOutputBindingInfo("Placeholder").second.GetNumDimensions() == 2);
    BOOST_TEST(m_Parser->GetNetworkOutputBindingInfo("Placeholder").second.GetShape()[0] == 1);
    BOOST_TEST(m_Parser->GetNetworkOutputBindingInfo("Placeholder").second.GetShape()[1] == 7);
}

BOOST_FIXTURE_TEST_CASE(RunGraph, PassThruFixture)
{
    armnn::TensorInfo inputTensorInfo = m_Parser->GetNetworkInputBindingInfo("Placeholder").second;
    auto input = MakeRandomTensor<float, 2>(inputTensorInfo, 378346);
    std::vector<float> inputVec;
    inputVec.assign(input.data(), input.data() + input.num_elements());
    RunTest<2>(inputVec, inputVec); // The passthru network should output the same as the input.
}

BOOST_AUTO_TEST_SUITE_END()
