//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct ActivationFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    explicit ActivationFixture(const char* activationFunction)
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
            "        unknown_rank: true\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "}\n"
            "node {\n"
            "  name: \"";
        m_Prototext.append(activationFunction);
        m_Prototext.append("\"\n"
                               "  op: \"");
        m_Prototext.append(activationFunction);
        m_Prototext.append("\"\n"
                               "  input: \"Placeholder\"\n"
                               "  attr {\n"
                               "    key: \"T\"\n"
                               "    value {\n"
                               "      type: DT_FLOAT\n"
                               "    }\n"
                               "  }\n"
                               "}\n");

        SetupSingleInputSingleOutput({ 1, 7 }, "Placeholder", activationFunction);
    }
};


struct ReLuFixture : ActivationFixture
{
    ReLuFixture() : ActivationFixture("Relu") {}
};
BOOST_FIXTURE_TEST_CASE(ParseReLu, ReLuFixture)
{
    RunTest<2>({ -1.0f, -0.5f, 1.25f, -3.0f, 0.0f, 0.5f, -0.75f },
               { 0.0f, 0.0f, 1.25f, 0.0f, 0.0f, 0.5f, 0.0f });
}


struct ReLu6Fixture : ActivationFixture
{
    ReLu6Fixture() : ActivationFixture("Relu6") {}
};
BOOST_FIXTURE_TEST_CASE(ParseReLu6, ReLu6Fixture)
{
    RunTest<2>({ -1.0f, -0.5f, 7.25f, -3.0f, 0.0f, 0.5f, -0.75f },
               { 0.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.5f, 0.0f });
}


struct SigmoidFixture : ActivationFixture
{
    SigmoidFixture() : ActivationFixture("Sigmoid") {}
};
BOOST_FIXTURE_TEST_CASE(ParseSigmoid, SigmoidFixture)
{
    RunTest<2>({ -0.1f, -0.2f, -0.3f, -0.4f, 0.1f, 0.2f, 0.3f },
               { 0.4750208f, 0.45016602f, 0.42555749f, 0.40131235f, 0.52497917f, 0.54983395f, 0.57444251f });
}


struct SoftplusFixture : ActivationFixture
{
    SoftplusFixture() : ActivationFixture("Softplus") {}
};
BOOST_FIXTURE_TEST_CASE(ParseSoftplus, SoftplusFixture)
{
    RunTest<2>({ -0.1f, -0.2f, -0.3f, -0.4f, 0.1f, 0.2f, 0.3f },
               { 0.64439666f, 0.59813893f, 0.55435526f, 0.51301527f, 0.74439669f, 0.7981388f, 0.85435522f });
}


struct TanhFixture : ActivationFixture
{
    TanhFixture() : ActivationFixture("Tanh") {}
};
BOOST_FIXTURE_TEST_CASE(ParseTanh, TanhFixture)
{
    RunTest<2>({ -0.1f, -0.2f, -0.3f, -0.4f, 0.1f, 0.2f, 0.3f },
               { -0.09966799f, -0.19737528f, -0.29131261f, -0.379949f, 0.09966799f, 0.19737528f, 0.29131261f });
}

BOOST_AUTO_TEST_SUITE_END()
