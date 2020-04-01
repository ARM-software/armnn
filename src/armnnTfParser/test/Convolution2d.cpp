//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

#include <array>
#include <string>
#include <iostream>

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct Convolution2dFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    explicit Convolution2dFixture(const std::string& dataLayout, const std::string& paddingType)
    : Convolution2dFixture(dataLayout, paddingType, 1)
    {}

    // Dilation: 0 - dilations attribute is not included;
    // Dilation: >0 - dilations attribute set to [1,v,v,1], where v is the value of the dilation arg
    explicit Convolution2dFixture(const std::string& dataLayout, const std::string& paddingType,
                                  int stride, int dilation = 0)
    {
        std::string strideString ("        i: 1 \n"
                                  "        i: 1 \n");
        if (dataLayout == "NHWC")
        {
            strideString.append("        i: " + std::to_string(stride) + " \n"
                                "        i: 1 \n");
        }
        else // dataLayout == "NCHW"
        {
            strideString.append("        i: 1 \n"
                                "        i: " + std::to_string(stride) + " \n");
        }

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
                           "      list { \n");
        m_Prototext.append(strideString);

        m_Prototext.append("      } \n"
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
        ARMNN_ASSERT_MSG(stride == 1 || stride == 2, "Add support for strides other than 1 or 2.");
        std::array<unsigned int, 4> dims;
        if (dataLayout == "NHWC")
        {
            dims = { 1u, (stride == 2 ? 3u : 2u), 3u, 1u };
        }
        else // dataLayout == "NCHW"
        {
            dims = { 1u, 1u, (stride == 2 ? 3u : 2u), 3u };
        }

        SetupSingleInputSingleOutput(armnn::TensorShape(4, dims.data()), "graphInput", "potato");
    }
};


struct Convolution2dNhwcSameFixture : Convolution2dFixture
{
    Convolution2dNhwcSameFixture() : Convolution2dFixture("NHWC", "SAME", 1){}
};
BOOST_FIXTURE_TEST_CASE(ParseConv2dNhwcSame, Convolution2dNhwcSameFixture)
{
    RunTest<4>({1, 2, 3, 4, 5, 6}, {2, 4, 4, 6.5f, 10 , 8.5f});
}

struct Convolution2dNchwSameFixture : Convolution2dFixture
{
    Convolution2dNchwSameFixture() : Convolution2dFixture("NCHW", "SAME", 1){}
};
BOOST_FIXTURE_TEST_CASE(ParseConv2dNchwSame, Convolution2dNchwSameFixture)
{
    RunTest<4>({1, 2, 3, 4, 5, 6}, {2, 4, 4, 6.5f, 10 , 8.5f});
}


struct Convolution2dNhwcValidFixture : Convolution2dFixture
{
    Convolution2dNhwcValidFixture() : Convolution2dFixture("NHWC", "VALID", 1){}
};
BOOST_FIXTURE_TEST_CASE(ParseConv2dNhwcValid, Convolution2dNhwcValidFixture)
{
    RunTest<4>({1, 2, 3, 4, 5, 6}, {4, 10});
}

struct Convolution2dNchwValidFixture : Convolution2dFixture
{
    Convolution2dNchwValidFixture() : Convolution2dFixture("NCHW", "VALID", 1){}
};
BOOST_FIXTURE_TEST_CASE(ParseConv2dNchwValid, Convolution2dNchwValidFixture)
{
    RunTest<4>({1, 2, 3, 4, 5, 6}, {4, 10});
}


struct Convolution2dStride2NhwcSameFixture : Convolution2dFixture
{
    Convolution2dStride2NhwcSameFixture() : Convolution2dFixture("NHWC", "SAME", 2){}
};
BOOST_FIXTURE_TEST_CASE(ParseConv2dStride2NhwcSame, Convolution2dStride2NhwcSameFixture)
{
    RunTest<4>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {2, 4, 6.5, 8.5, 11, 13});
}

struct Convolution2dStride2NchwSameFixture : Convolution2dFixture
{
    Convolution2dStride2NchwSameFixture() : Convolution2dFixture("NCHW", "SAME", 2){}
};
BOOST_FIXTURE_TEST_CASE(ParseConv2dStride2NchwSame, Convolution2dStride2NchwSameFixture)
{
    RunTest<4>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {2, 4, 6.5, 8.5, 11, 13});
}


struct Convolution2dStride2NhwcValidFixture : Convolution2dFixture
{
    Convolution2dStride2NhwcValidFixture() : Convolution2dFixture("NHWC", "VALID", 2){}
};
BOOST_FIXTURE_TEST_CASE(ParseConv2dStride2NhwcValid, Convolution2dStride2NhwcValidFixture)
{
    RunTest<4>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {4, 10, 16});
}

struct Convolution2dStride2NchwValidFixture : Convolution2dFixture
{
    Convolution2dStride2NchwValidFixture() : Convolution2dFixture("NCHW", "VALID", 2){}
};
BOOST_FIXTURE_TEST_CASE(ParseConv2dStride2NchwValid, Convolution2dStride2NchwValidFixture)
{
    RunTest<4>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {4, 10, 16});
}


struct Convolution2dDilation1NhwcFixture : Convolution2dFixture
{
    Convolution2dDilation1NhwcFixture() : Convolution2dFixture("NHWC", "SAME", 1, 1){}
};
BOOST_FIXTURE_TEST_CASE(ParseConv2dDilation1Nhwc, Convolution2dDilation1NhwcFixture)
{
    RunTest<4>({1, 2, 3, 4, 5, 6}, {2, 4, 4, 6.5f, 10 , 8.5f});
}

struct Convolution2dDilation1NchwFixture : Convolution2dFixture
{
    Convolution2dDilation1NchwFixture() : Convolution2dFixture("NCHW", "SAME", 1, 1){}
};
BOOST_FIXTURE_TEST_CASE(ParseConv2dDilation1Nchw, Convolution2dDilation1NchwFixture)
{
    RunTest<4>({1, 2, 3, 4, 5, 6}, {2, 4, 4, 6.5f, 10 , 8.5f});
}


BOOST_AUTO_TEST_CASE(ParseConv2dDilation2)
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
    BOOST_CHECK_THROW(parser->CreateNetworkFromString(prototext, inputShapes, { "potato" }), armnn::ParseException);
}


BOOST_AUTO_TEST_SUITE_END()
