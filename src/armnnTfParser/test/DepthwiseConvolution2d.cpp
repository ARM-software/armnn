//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"
#include <string>
#include <iostream>

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct DepthwiseConvolution2dFixture : public ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    explicit DepthwiseConvolution2dFixture(const char* paddingType)
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
                      "      key: \"value\" \n"
                      "      value { \n"
                      "        tensor { \n"
                      "          dtype: DT_FLOAT \n"
                      "          tensor_shape { \n"
                      "            dim { \n"
                      "              size: 1 \n"
                      "            } \n"
                      "            dim { \n"
                      "              size: 1 \n"
                      "            } \n"
                      "            dim { \n"
                      "              size: 3 \n"
                      "            } \n"
                      "            dim { \n"
                      "              size: 3 \n"
                      "            } \n"
                      "          } \n"
                      "          tensor_content: \"\\000\\000\\200?\\000\\000\\000@\\000\\000@@\\000\\000\\200@"
                      "\\000\\000\\240@\\000\\000\\300@\\000\\000\\340@\\000\\000\\000A\\000\\000\\020A\" \n"
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
                      "      s: \"NHWC\" \n"
                      "    } \n"
                      "  } \n"
                      "  attr { \n"
                      "    key: \"padding\" \n"
                      "    value { \n"
                      "      s: \"";
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

        SetupSingleInputSingleOutput({ 1, 1, 3, 3 }, "graphInput", "potato");
    }
};

struct DepthwiseConvolution2dSameFixture : DepthwiseConvolution2dFixture
{
    DepthwiseConvolution2dSameFixture() : DepthwiseConvolution2dFixture("SAME") { }
};

BOOST_FIXTURE_TEST_CASE(ParseDepthwiseConv2DSame, DepthwiseConvolution2dSameFixture)
{
    RunTest<4>({ 1, 2, 3, 4, 5, 6, 7, 8, 9 },
               { 2.5f, 5.f,  2.5f, 3.5f, 7.f,  3.5f, 4.5f, 9.f,  4.5f,
                 6.f,  12.f, 6.f,  7.5f, 15.f, 7.5f, 9.f,  18.f, 9.f,
                 5.5f, 11.f, 5.5f, 6.5f, 13.f, 6.5f, 7.5f, 15.f, 7.5f});
}

struct DepthwiseConvolution2dValidFixture : DepthwiseConvolution2dFixture
{
    DepthwiseConvolution2dValidFixture() : DepthwiseConvolution2dFixture("VALID") { }
};

BOOST_FIXTURE_TEST_CASE(ParseDepthwiseConv2DValid, DepthwiseConvolution2dValidFixture)
{
    RunTest<4>({ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, // input data
               { 6.f,  12.f, 6.f,  7.5f, 15.f, 7.5f, 9.f,  18.f, 9.f });  // output expected data
}


BOOST_AUTO_TEST_SUITE_END()
