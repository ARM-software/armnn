//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserPrototxtFixture.hpp"

#include "armnnTfParser/ITfParser.hpp"

#include <armnnUtils/Permute.hpp>

#include <boost/test/unit_test.hpp>

#include <string>
#include <iostream>

using namespace armnnUtils;
using namespace armnn;

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct DepthwiseConvolution2dFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    explicit DepthwiseConvolution2dFixture(const std::string& dataLayout, const char* paddingType)
    {
        m_Prototext = "node { \n"
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
                      "  node { \n"
                      "  name: \"Const_1\" \n"
                      "  op: \"Const\" \n"
                      "  attr { \n"
                      "    key: \"dtype\" \n"
                      "    value { \n"
                      "      type: DT_FLOAT \n"
                      "    } \n"
                      "  } \n"
                      "  attr { \n"
                      "    key: \"value\" \n"
                      "    value { \n"
                      "      tensor { \n"
                      "        dtype: DT_FLOAT \n"
                      "        tensor_shape { \n"
                      "          dim { \n"
                      "            size: 1 \n"
                      "          } \n"
                      "          dim { \n"
                      "            size: 3 \n"
                      "          } \n"
                      "          dim { \n"
                      "            size: 3 \n"
                      "          } \n"
                      "          dim { \n"
                      "            size: 3 \n"
                      "          } \n"
                      "        } \n"
                      "        tensor_content: \"\\000\\000\\000?\\000\\000\\200?\\000\\000\\000?"
                      "\\000\\000\\000?\\000\\000\\200?\\000\\000\\000?"
                      "\\000\\000\\000?\\000\\000\\200?\\000\\000\\000?"
                      "\\000\\000\\000?\\000\\000\\200?\\000\\000\\000?"
                      "\\000\\000\\000?\\000\\000\\200?\\000\\000\\000?"
                      "\\000\\000\\000?\\000\\000\\200?\\000\\000\\000?"
                      "\\000\\000\\000?\\000\\000\\200?\\000\\000\\000?"
                      "\\000\\000\\000?\\000\\000\\200?\\000\\000\\000?"
                      "\\000\\000\\000?\\000\\000\\200?\\000\\000\\000?\" \n"
                      "      } \n"
                      "    } \n"
                      "  } \n"
                      "} \n"
                      "node { \n"
                      "  name: \"potato\" \n"
                      "  op: \"DepthwiseConv2dNative\" \n"
                      "  input: \"graphInput\" \n"
                      "  input: \"Const_1\" \n"
                      "  attr { \n"
                      "    key: \"T\" \n"
                      "    value { \n"
                      "      type: DT_FLOAT \n"
                      "    } \n"
                      "  } \n"
                      "  attr { \n"
                      "    key: \"data_format\" \n"
                      "    value { \n"
                      "      s: \"";
        m_Prototext.append(dataLayout);
        m_Prototext.append("\"\n"
                      "    } \n"
                      "  } \n"
                      "  attr { \n"
                      "    key: \"padding\" \n"
                      "    value { \n"
                      "      s: \"");
        m_Prototext.append(paddingType);
        m_Prototext.append("\"\n"
                      "    } \n"
                      "  } \n"
                      "  attr { \n"
                      "    key: \"strides\" \n"
                      "    value { \n"
                      "      list { \n"
                      "        i: 1 \n"
                      "        i: 1 \n"
                      "        i: 1 \n"
                      "        i: 1 \n"
                      "      } \n"
                      "    } \n"
                      "  } \n"
                      "  attr { \n"
                      "    key: \"use_cudnn_on_gpu\" \n"
                      "    value { \n"
                      "      b: false \n"
                      "    } \n"
                      "  } \n"
                      "} \n");

        if(dataLayout == "NHWC")
        {
            SetupSingleInputSingleOutput({ 1u, 1u, 3u, 3u }, "graphInput", "potato");
        }
        else
        {
            SetupSingleInputSingleOutput({ 1u, 3u, 1u, 3u }, "graphInput", "potato");
        }
    }
};

struct DepthwiseConvolution2dNhwcSameFixture : DepthwiseConvolution2dFixture
{
    DepthwiseConvolution2dNhwcSameFixture() : DepthwiseConvolution2dFixture("NHWC", "SAME") { }
};

BOOST_FIXTURE_TEST_CASE(ParseDepthwiseConv2DNhwcSame, DepthwiseConvolution2dNhwcSameFixture)
{
    RunTest<4>({ 1, 2, 3, 4, 5, 6, 7, 8, 9 },
               { 2.5f, 5.f,  2.5f, 3.5f, 7.f,  3.5f, 4.5f, 9.f,  4.5f,
                 6.f,  12.f, 6.f,  7.5f, 15.f, 7.5f, 9.f,  18.f, 9.f,
                 5.5f, 11.f, 5.5f, 6.5f, 13.f, 6.5f, 7.5f, 15.f, 7.5f });
}

struct DepthwiseConvolution2dNchwSameFixture : DepthwiseConvolution2dFixture
{
    DepthwiseConvolution2dNchwSameFixture() : DepthwiseConvolution2dFixture("NCHW", "SAME") { }
};

BOOST_FIXTURE_TEST_CASE(ParseDepthwiseConv2DNchwSame, DepthwiseConvolution2dNchwSameFixture)
{
    RunTest<4>({ 1, 4, 7, 2, 5, 8, 3, 6, 9 },
               { 2.5f, 6.f, 5.5f, 5.f, 12.f, 11.f, 2.5f, 6.f, 5.5f,
                 3.5f, 7.5f, 6.5f, 7.f, 15.f, 13.f, 3.5f, 7.5f, 6.5f,
                 4.5f, 9.f, 7.5f, 9.f, 18.f, 15.f, 4.5f, 9.f, 7.5f });
}

struct DepthwiseConvolution2dNhwcValidFixture : DepthwiseConvolution2dFixture
{
    DepthwiseConvolution2dNhwcValidFixture() : DepthwiseConvolution2dFixture("NHWC", "VALID") { }
};

BOOST_FIXTURE_TEST_CASE(ParseDepthwiseConv2DNhwcValid, DepthwiseConvolution2dNhwcValidFixture)
{
    RunTest<4>({ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, // input data
               { 6.f, 12.f, 6.f, 7.5f, 15.f, 7.5f, 9.f, 18.f, 9.f });  // output expected data
}

struct DepthwiseConvolution2dNchwValidFixture : DepthwiseConvolution2dFixture
{
    DepthwiseConvolution2dNchwValidFixture() : DepthwiseConvolution2dFixture("NCHW", "VALID") { }
};

BOOST_FIXTURE_TEST_CASE(ParseDepthwiseConv2DNchwValid, DepthwiseConvolution2dNchwValidFixture)
{
     RunTest<4>({ 1, 4, 7, 2, 5, 8, 3, 6, 9 },
                { 6.f, 12.f, 6.f, 7.5f, 15.f, 7.5f, 9.f, 18.f, 9.f });
}


BOOST_AUTO_TEST_SUITE_END()
