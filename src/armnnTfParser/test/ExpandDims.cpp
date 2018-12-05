//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct ExpandDimsFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    ExpandDimsFixture(const std::string& expandDim)
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
                "  name: \"ExpandDims\" \n"
                "  op: \"ExpandDims\" \n"
                "  input: \"graphInput\" \n"
                "  attr { \n"
                "    key: \"T\" \n"
                "    value { \n"
                "      type: DT_FLOAT \n"
                "    } \n"
                "  } \n"
                "  attr { \n"
                "    key: \"Tdim\" \n"
                "    value { \n";
            m_Prototext += "i:" + expandDim;
            m_Prototext +=
                "    } \n"
                "  } \n"
                "} \n";

        SetupSingleInputSingleOutput({ 2, 3, 5 }, "graphInput", "ExpandDims");
    }
};

struct ExpandZeroDim : ExpandDimsFixture
{
    ExpandZeroDim() : ExpandDimsFixture("0") {}
};

struct ExpandTwoDim : ExpandDimsFixture
{
    ExpandTwoDim() : ExpandDimsFixture("2") {}
};

struct ExpandThreeDim : ExpandDimsFixture
{
    ExpandThreeDim() : ExpandDimsFixture("3") {}
};

struct ExpandMinusOneDim : ExpandDimsFixture
{
    ExpandMinusOneDim() : ExpandDimsFixture("-1") {}
};

struct ExpandMinusThreeDim : ExpandDimsFixture
{
    ExpandMinusThreeDim() : ExpandDimsFixture("-3") {}
};

BOOST_FIXTURE_TEST_CASE(ParseExpandZeroDim, ExpandZeroDim)
{
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo("ExpandDims").second.GetShape() ==
                armnn::TensorShape({1, 2, 3, 5})));
}

BOOST_FIXTURE_TEST_CASE(ParseExpandTwoDim, ExpandTwoDim)
{
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo("ExpandDims").second.GetShape() ==
                armnn::TensorShape({2, 3, 1, 5})));
}

BOOST_FIXTURE_TEST_CASE(ParseExpandThreeDim, ExpandThreeDim)
{
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo("ExpandDims").second.GetShape() ==
                armnn::TensorShape({2, 3, 5, 1})));
}

BOOST_FIXTURE_TEST_CASE(ParseExpandMinusOneDim, ExpandMinusOneDim)
{
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo("ExpandDims").second.GetShape() ==
                armnn::TensorShape({2, 3, 5, 1})));
}

BOOST_FIXTURE_TEST_CASE(ParseExpandMinusThreeDim, ExpandMinusThreeDim)
{
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo("ExpandDims").second.GetShape() ==
                armnn::TensorShape({2, 1, 3, 5})));
}

BOOST_AUTO_TEST_SUITE_END()
