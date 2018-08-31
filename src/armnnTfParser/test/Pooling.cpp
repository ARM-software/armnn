//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct Pooling2dFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    explicit Pooling2dFixture(const char* poolingtype)
    {
        m_Prototext =  "node {\n"
            "  name: \"Placeholder\"\n"
            "  op: \"Placeholder\"\n"
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
            "        }\n"
            "      }\n"
            "    }\n"
            "   }\n"
            "  }\n"
            "node {\n"
            "  name: \"";
        m_Prototext.append(poolingtype);
        m_Prototext.append("\"\n"
                               "  op: \"");
        m_Prototext.append(poolingtype);
        m_Prototext.append("\"\n"
                               "  input: \"Placeholder\"\n"
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
                               "    key: \"ksize\"\n"
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
                               "    key: \"padding\"\n"
                               "    value {\n"
                               "      s: \"VALID\"\n"
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
                               "}\n");

        SetupSingleInputSingleOutput({ 1, 2, 2, 1 }, "Placeholder", poolingtype);
    }
};


struct MaxPoolFixture : Pooling2dFixture
{
    MaxPoolFixture() : Pooling2dFixture("MaxPool") {}
};
BOOST_FIXTURE_TEST_CASE(ParseMaxPool, MaxPoolFixture)
{
    RunTest<4>({1.0f, 2.0f, 3.0f, -4.0f}, {3.0f});
}


struct AvgPoolFixture : Pooling2dFixture
{
    AvgPoolFixture() : Pooling2dFixture("AvgPool") {}
};
BOOST_FIXTURE_TEST_CASE(ParseAvgPool, AvgPoolFixture)
{
    RunTest<4>({1.0f, 2.0f, 3.0f, 4.0f}, {2.5f});
}


BOOST_AUTO_TEST_SUITE_END()
