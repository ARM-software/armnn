//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct Pooling2dFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    explicit Pooling2dFixture(const char* poolingtype, std::string dataLayout, std::string paddingOption)
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
                               "      s: \"");
        m_Prototext.append(dataLayout);
        m_Prototext.append("\"\n"
                               "    }\n"
                               "  }\n"
                               "  attr {\n"
                               "    key: \"ksize\"\n"
                               "    value {\n"
                               "      list {\n"

                               "        i: 1\n");
        if(dataLayout == "NHWC")
        {
            m_Prototext.append("        i: 2\n"
                               "        i: 2\n"
                               "        i: 1\n");
        }
        else
        {
            m_Prototext.append("        i: 1\n"
                               "        i: 2\n"
                               "        i: 2\n");
        }
        m_Prototext.append(
                               "      }\n"
                               "    }\n"
                               "  }\n"
                               "  attr {\n"
                               "    key: \"padding\"\n"
                               "    value {\n"
                               "      s: \"");
        m_Prototext.append(paddingOption);
        m_Prototext.append(
                               "\"\n"
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

        if(dataLayout == "NHWC")
        {
            SetupSingleInputSingleOutput({ 1, 2, 2, 1 }, "Placeholder", poolingtype);
        }
        else
        {
            SetupSingleInputSingleOutput({ 1, 1, 2, 2 }, "Placeholder", poolingtype);
        }
    }
};


struct MaxPoolFixtureNhwcValid : Pooling2dFixture
{
    MaxPoolFixtureNhwcValid() : Pooling2dFixture("MaxPool", "NHWC", "VALID") {}
};
BOOST_FIXTURE_TEST_CASE(ParseMaxPoolNhwcValid, MaxPoolFixtureNhwcValid)
{
    RunTest<4>({1.0f, 2.0f, 3.0f, -4.0f}, {3.0f});
}

struct MaxPoolFixtureNchwValid : Pooling2dFixture
{
    MaxPoolFixtureNchwValid() : Pooling2dFixture("MaxPool", "NCHW", "VALID") {}
};
BOOST_FIXTURE_TEST_CASE(ParseMaxPoolNchwValid, MaxPoolFixtureNchwValid)
{
    RunTest<4>({1.0f, 2.0f, 3.0f, -4.0f}, {3.0f});
}

struct MaxPoolFixtureNhwcSame : Pooling2dFixture
{
    MaxPoolFixtureNhwcSame() : Pooling2dFixture("MaxPool", "NHWC", "SAME") {}
};
BOOST_FIXTURE_TEST_CASE(ParseMaxPoolNhwcSame, MaxPoolFixtureNhwcSame)
{
    RunTest<4>({1.0f, 2.0f, 3.0f, -4.0f}, {3.0f, 2.0f, 3.0f, -4.0f});
}

struct MaxPoolFixtureNchwSame : Pooling2dFixture
{
    MaxPoolFixtureNchwSame() : Pooling2dFixture("MaxPool", "NCHW", "SAME") {}
};
BOOST_FIXTURE_TEST_CASE(ParseMaxPoolNchwSame, MaxPoolFixtureNchwSame)
{
    RunTest<4>({1.0f, 2.0f, 3.0f, -4.0f}, {3.0f, 2.0f, 3.0f, -4.0f});
}

struct AvgPoolFixtureNhwcValid : Pooling2dFixture
{
    AvgPoolFixtureNhwcValid() : Pooling2dFixture("AvgPool", "NHWC", "VALID") {}
};
BOOST_FIXTURE_TEST_CASE(ParseAvgPoolNhwcValid, AvgPoolFixtureNhwcValid)
{
    RunTest<4>({1.0f, 2.0f, 3.0f, 4.0f}, {2.5f});
}

struct AvgPoolFixtureNchwValid : Pooling2dFixture
{
    AvgPoolFixtureNchwValid() : Pooling2dFixture("AvgPool", "NCHW", "VALID") {}
};
BOOST_FIXTURE_TEST_CASE(ParseAvgPoolNchwValid, AvgPoolFixtureNchwValid)
{
    RunTest<4>({1.0f, 2.0f, 3.0f, 4.0f}, {2.5f});
}

struct AvgPoolFixtureNhwcSame : Pooling2dFixture
{
    AvgPoolFixtureNhwcSame() : Pooling2dFixture("AvgPool", "NHWC", "SAME") {}
};
BOOST_FIXTURE_TEST_CASE(ParseAvgPoolNhwcSame, AvgPoolFixtureNhwcSame)
{
    RunTest<4>({1.0f, 2.0f, 3.0f, 4.0f}, {2.5f, 3.0f, 3.5f, 4.0f});
}

struct AvgPoolFixtureNchwSame : Pooling2dFixture
{
    AvgPoolFixtureNchwSame() : Pooling2dFixture("AvgPool", "NCHW", "SAME") {}
};
BOOST_FIXTURE_TEST_CASE(ParseAvgPoolNchwSame, AvgPoolFixtureNchwSame)
{
    RunTest<4>({1.0f, 2.0f, 3.0f, 4.0f}, {2.5f, 3.0f, 3.5f, 4.0f});
}

BOOST_AUTO_TEST_SUITE_END()
