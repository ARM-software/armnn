//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"
#include <string>
#include <iostream>

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct Convolution2dFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    explicit Convolution2dFixture(const char* paddingType)
    : Convolution2dFixture(paddingType, 1)
    {}

    // Dilation: 0 - dilations attribute is not included;
    // Dilation: >0 - dilations attribute set to [1,v,v,1], where v is the value of the dilation arg
    explicit Convolution2dFixture(const char* paddingType, int stride, int dilation = 0)
    {
        std::string strideString = std::to_string(stride);
        std::string dilationString = std::to_string(dilation);
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
            "            size: 1 \n"
            "          } \n"
            "          dim { \n"
            "            size: 1 \n"
            "          } \n"
            "        } \n"
            "        tensor_content: \"\\000\\000\\000?\\000\\000\\200?\\000\\000\\000?\" \n"
            "      } \n"
            "    } \n"
            "  } \n"
            "} \n"
            "node { \n"
            "  name: \"potato\" \n"
            "  op: \"Conv2D\" \n"
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
                           "        i: ");
        m_Prototext.append(strideString);
        m_Prototext.append(" \n"
                           "        i: 1 \n"
                           "      } \n"
                           "    } \n"
                           "  } \n");

        if (dilation > 0)
        {
            m_Prototext.append("  attr { \n"
                               "    key: \"dilations\" \n"
                               "    value { \n"
                               "      list { \n"
                               "        i: 1 \n"
                               "        i: ");
            m_Prototext.append(dilationString);
            m_Prototext.append(" \n"
                               "        i: ");
            m_Prototext.append(dilationString);
            m_Prototext.append(" \n"
                               "        i: 1 \n"
                               "      } \n"
                               "    } \n"
                               "  } \n");
        }
        m_Prototext.append("  attr { \n"
                           "    key: \"use_cudnn_on_gpu\" \n"
                           "    value { \n"
                           "      b: false \n"
                           "    } \n"
                           "  } \n"
                           "} \n");

        // Manual height computation based on stride parameter.
        BOOST_ASSERT_MSG(stride == 1 || stride==2, "Add support for strides other than 1 or 2.");
        unsigned int dims[] = {1,2,3,1};
        if (stride == 2)
        {
            dims[1]=3;
        }

        SetupSingleInputSingleOutput(armnn::TensorShape(4, dims), "graphInput", "potato");
    }
};


struct Convolution2dSameFixture : Convolution2dFixture
{
    Convolution2dSameFixture() : Convolution2dFixture("SAME", 1){}
};
BOOST_FIXTURE_TEST_CASE(ParseConv2DSame, Convolution2dSameFixture)
{
    RunTest<4>({1, 2, 3, 4, 5, 6}, {2, 4, 4, 6.5f, 10 , 8.5f});
}

struct Convolution2dValidFixture : Convolution2dFixture
{
    Convolution2dValidFixture() : Convolution2dFixture("VALID", 1){}
};
BOOST_FIXTURE_TEST_CASE(ParseConv2DValid, Convolution2dValidFixture)
{
    RunTest<4>({1, 2, 3, 4, 5, 6}, {4, 10});
}


struct Convolution2dStride2SameFixture : Convolution2dFixture
{
    Convolution2dStride2SameFixture() : Convolution2dFixture("SAME", 2){}
};
BOOST_FIXTURE_TEST_CASE(ParseConv2DStride2Same, Convolution2dStride2SameFixture)
{
    RunTest<4>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {2, 4, 6.5, 8.5, 11, 13});
}


struct Convolution2dStride2ValidFixture : Convolution2dFixture
{
    Convolution2dStride2ValidFixture() : Convolution2dFixture("VALID", 2){}
};
BOOST_FIXTURE_TEST_CASE(ParseConv2DStride2Valid, Convolution2dStride2ValidFixture)
{
    RunTest<4>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {4, 10, 16});
}


struct Convolution2dDilation1Fixture : Convolution2dFixture
{
    Convolution2dDilation1Fixture() : Convolution2dFixture("SAME", 1, 1){}
};
BOOST_FIXTURE_TEST_CASE(ParseConv2DDilation1, Convolution2dDilation1Fixture)
{
    RunTest<4>({1, 2, 3, 4, 5, 6}, {2, 4, 4, 6.5f, 10 , 8.5f});
}

BOOST_AUTO_TEST_CASE(ParseConv2DDilation2)
{
    const char* prototext = ""
        "node {\n"
        "  name: \"graphInput\"\n"
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
        "}\n"
        "node {\n"
        "  name: \"Const_1\"\n"
        "  op: \"Const\"\n"
        "  attr {\n"
        "    key: \"dtype\"\n"
        "    value {\n"
        "      type: DT_FLOAT\n"
        "    }\n"
        "  }\n"
        "  attr {\n"
        "    key: \"value\"\n"
        "    value {\n"
        "      tensor {\n"
        "        dtype: DT_FLOAT\n"
        "        tensor_shape {\n"
        "          dim {\n"
        "            size: 1\n"
        "          }\n"
        "          dim {\n"
        "            size: 3\n"
        "          }\n"
        "          dim {\n"
        "            size: 1\n"
        "          }\n"
        "          dim {\n"
        "            size: 1\n"
        "          }\n"
        "        }\n"
        "        tensor_content: \"\\000\\000\\000?\\000\\000\\200?\\000\\000\\000?\"\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n"
        "node {\n"
        "  name: \"potato\"\n"
        "  op: \"Conv2D\"\n"
        "  input: \"graphInput\"\n"
        "  input: \"Const_1\"\n"
        "  attr {\n"
        "    key: \"T\"\n"
        "    value {\n"
        "      type: DT_FLOAT\n"
        "    }\n"
        "  }\n"
        "  attr {\n"
        "    key: \"data_format\"\n"
        "    value {\n"
        "      s: \"NHWC\"\n"
        "    }\n"
        "  }\n"
        "  attr {\n"
        "    key: \"padding\"\n"
        "    value {\n"
        "      s: \"SAME\"\n"
        "    }\n"
        "  }\n"
        "  attr {\n"
        "    key: \"strides\"\n"
        "    value {\n"
        "      list {\n"
        "        i: 1\n"
        "        i: 1\n"
        "        i: 1\n"
        "        i: 1\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "  attr {\n"
        "    key: \"dilations\"\n"
        "    value {\n"
        "      list {\n"
        "        i: 1\n"
        "        i: 2\n"
        "        i: 2\n"
        "        i: 1\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "  attr {\n"
        "    key: \"use_cudnn_on_gpu\"\n"
        "    value {\n"
        "      b: false\n"
        "    }\n"
        "  }\n"
        "}\n";

    std::map<std::string, armnn::TensorShape> inputShapes;
    armnn::TensorShape tensorShape = { 1, 3, 3, 1 };
    inputShapes["graphInput"] = tensorShape;
    armnnTfParser::ITfParserPtr parser = armnnTfParser::ITfParser::Create();
    BOOST_CHECK_THROW(parser->CreateNetworkFromString(prototext, inputShapes, { "potato" }),
                          armnn::ParseException);
}


BOOST_AUTO_TEST_SUITE_END()
