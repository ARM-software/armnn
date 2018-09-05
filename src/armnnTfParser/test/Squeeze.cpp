//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

template <bool withDimZero, bool withDimOne>
struct SqueezeFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    SqueezeFixture()
    {
        m_Prototext =
                "node { \n"
                "    name: \"graphInput\" \n"
                "    op: \"Placeholder\" \n"
                "    attr { \n"
                "      key: \"dtype\" \n"
                "      value { \n"
                "        type: DT_FLOAT \n"
                "      } \n"
                "    } \n"
                "    attr { \n"
                "      key: \"shape\" \n"
                "      value { \n"
                "        shape { \n"
                "        } \n"
                "      } \n"
                "    } \n"
                "  } \n"
                "node { \n"
                "  name: \"Squeeze\" \n"
                "  op: \"Squeeze\" \n"
                "  input: \"graphInput\" \n"
                "  attr { \n"
                "    key: \"T\" \n"
                "    value { \n"
                "      type: DT_FLOAT \n"
                "    } \n"
                "  } \n"
                "  attr { \n"
                "    key: \"squeeze_dims\" \n"
                "    value { \n"
                "      list {\n";

        if (withDimZero)
        {
            m_Prototext += "i:0\n";
        }

        if (withDimOne)
        {
            m_Prototext += "i:1\n";
        }

        m_Prototext +=
                "      } \n"
                "    } \n"
                "  } \n"
                "} \n";

        SetupSingleInputSingleOutput({ 1, 1, 2, 2 }, "graphInput", "Squeeze");
    }
};

typedef SqueezeFixture<false, false> ImpliedDimensionsSqueezeFixture;
typedef SqueezeFixture<true, false>  ExplicitDimensionZeroSqueezeFixture;
typedef SqueezeFixture<false, true>  ExplicitDimensionOneSqueezeFixture;
typedef SqueezeFixture<true, true>   ExplicitDimensionsSqueezeFixture;

BOOST_FIXTURE_TEST_CASE(ParseImplicitSqueeze, ImpliedDimensionsSqueezeFixture)
{
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo("Squeeze").second.GetShape() ==
               armnn::TensorShape({2,2})));
    RunTest<2>({ 1.0f, 2.0f, 3.0f, 4.0f },
               { 1.0f, 2.0f, 3.0f, 4.0f });
}

BOOST_FIXTURE_TEST_CASE(ParseDimensionZeroSqueeze, ExplicitDimensionZeroSqueezeFixture)
{
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo("Squeeze").second.GetShape() ==
               armnn::TensorShape({1,2,2})));
    RunTest<3>({ 1.0f, 2.0f, 3.0f, 4.0f },
               { 1.0f, 2.0f, 3.0f, 4.0f });
}

BOOST_FIXTURE_TEST_CASE(ParseDimensionOneSqueeze, ExplicitDimensionOneSqueezeFixture)
{
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo("Squeeze").second.GetShape() ==
               armnn::TensorShape({1,2,2})));
    RunTest<3>({ 1.0f, 2.0f, 3.0f, 4.0f },
               { 1.0f, 2.0f, 3.0f, 4.0f });
}

BOOST_FIXTURE_TEST_CASE(ParseExplicitDimensionsSqueeze, ExplicitDimensionsSqueezeFixture)
{
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo("Squeeze").second.GetShape() ==
               armnn::TensorShape({2,2})));
    RunTest<2>({ 1.0f, 2.0f, 3.0f, 4.0f },
               { 1.0f, 2.0f, 3.0f, 4.0f });
}

BOOST_AUTO_TEST_SUITE_END()
