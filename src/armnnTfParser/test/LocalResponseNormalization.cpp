//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct LocalResponseNormalizationBaseFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    explicit LocalResponseNormalizationBaseFixture(float alpha, float beta, float bias)
    {
        std::string alphaString = std::to_string(alpha);
        std::string betaString = std::to_string(beta);
        std::string biasString = std::to_string(bias);

        m_Prototext = "node {"
            "  name: \"Placeholder\""
            "  op: \"Placeholder\""
            "  attr {"
            "    key: \"dtype\""
            "    value {"
            "      type: DT_FLOAT"
            "    }"
            "  }"
            "  attr {"
            "    key: \"shape\""
            "    value {"
            "      shape {"
            "        unknown_rank: true"
            "      }"
            "    }"
            "  }"
            "}"
            "node {"
            "  name: \"LRN\""
            "  op: \"LRN\""
            "  input: \"Placeholder\""
            "  attr {"
            "    key: \"T\""
            "    value {"
            "      type: DT_FLOAT"
            "    }"
            "  }"
            "  attr {"
            "    key: \"alpha\""
            "    value {"
            "      f: ";
        m_Prototext.append(alphaString);
        m_Prototext.append("\n"
            "    }"
            "  }"
            "  attr {"
            "    key: \"beta\""
            "    value {"
            "      f: ");
        m_Prototext.append(betaString);
        m_Prototext.append("\n"
            "    }"
            "  }"
            "  attr {"
            "    key: \"bias\""
            "    value {"
            "      f: ");
        m_Prototext.append(biasString);
        m_Prototext.append("\n"
            "    }"
            "  }"
            "  attr {"
            "    key: \"depth_radius\""
            "    value {"
            "      i: 1"
            "    }"
            "  }"
            "}");
    }
};


struct LocalResponseNormalizationFixtureSimple : public LocalResponseNormalizationBaseFixture
{
    explicit LocalResponseNormalizationFixtureSimple()
        : LocalResponseNormalizationBaseFixture(1.0f, 1.0f, 1.0f)
    {
        SetupSingleInputSingleOutput({ 2, 2, 2, 1 }, "Placeholder", "LRN");
    }
};
BOOST_FIXTURE_TEST_CASE(ParseSimpleLocalResponseNormalization, LocalResponseNormalizationFixtureSimple)
{
    RunTest<4>({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f },
               { 0.5f, 0.4f, 0.3f, 0.23529412f, 0.1923077f, 0.16216217f, 0.14f, 0.12307692f });
}


struct LocalResponseNormalizationFixture : public LocalResponseNormalizationBaseFixture
{
    explicit LocalResponseNormalizationFixture()
        : LocalResponseNormalizationBaseFixture(0.5f, 1.0f, 0.5f)
    {
        SetupSingleInputSingleOutput({1, 3, 3, 2}, "Placeholder", "LRN");
    }
};
BOOST_FIXTURE_TEST_CASE(ParseLocalResponseNormalization, LocalResponseNormalizationFixture)
{
    RunTest<4>({ 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                 7.0f,  8.0f,  9.0f, 10.0f, 11.0f, 12.0f,
                13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f},

               {0.333333340f, 0.66666670f, 0.230769250f, 0.307692320f, 0.161290320f, 0.19354838f,
                0.122807020f, 0.14035088f, 0.098901100f, 0.109890110f, 0.082706770f, 0.09022556f,
                0.071038246f, 0.07650273f, 0.062240668f, 0.066390045f, 0.055374593f, 0.05863192f});
}




BOOST_AUTO_TEST_SUITE_END()
