//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct PadFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    PadFixture() {
        m_Prototext = "node {\n"
                      "  name: \"input\"\n"
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
                      "        dim {\n"
                      "          size: -1\n"
                      "        }\n"
                      "        dim {\n"
                      "          size: 2\n"
                      "        }\n"
                      "        dim {\n"
                      "          size: 2\n"
                      "        }\n"
                      "        dim {\n"
                      "          size: 2\n"
                      "        }\n"
                      "      }\n"
                      "    }\n"
                      "  }\n"
                      "}\n"
                      "node {\n"
                      "  name: \"Pad/paddings\"\n"
                      "  op: \"Const\"\n"
                      "  attr {\n"
                      "    key: \"dtype\"\n"
                      "    value {\n"
                      "      type: DT_INT32\n"
                      "    }\n"
                      "  }\n"
                      "  attr {\n"
                      "    key: \"value\"\n"
                      "    value {\n"
                      "      tensor {\n"
                      "        dtype: DT_INT32\n"
                      "        tensor_shape {\n"
                      "          dim {\n"
                      "            size: 4\n"
                      "          }\n"
                      "          dim {\n"
                      "            size: 2\n"
                      "          }\n"
                      "        }\n"
                      "        tensor_content: \"\\000\\000\\000\\000\\000\\000\\000\\000"
                                                "\\001\\000\\000\\000\\001\\000\\000\\000"
                                                "\\001\\000\\000\\000\\001\\000\\000\\000"
                                                "\\000\\000\\000\\000\\000\\000\\000\\000\"\n"
                      "      }\n"
                      "    }\n"
                      "  }\n"
                      "}\n"
                      "node {\n"
                      "  name: \"Pad\"\n"
                      "  op: \"Pad\"\n"
                      "  input: \"input\"\n"
                      "  input: \"Pad/paddings\"\n"
                      "  attr {\n"
                      "    key: \"T\"\n"
                      "    value {\n"
                      "      type: DT_FLOAT\n"
                      "    }\n"
                      "  }\n"
                      "  attr {\n"
                      "    key: \"Tpaddings\"\n"
                      "    value {\n"
                      "      type: DT_INT32\n"
                      "    }\n"
                      "  }\n"
                      "}";

        SetupSingleInputSingleOutput({1, 2, 2, 2}, "input", "Pad");
    }
};

BOOST_FIXTURE_TEST_CASE(ParsePad, PadFixture)
{
    RunTest<4>({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f },
               { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 5.0f, 6.0f, 7.0f, 8.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
               });
}

BOOST_AUTO_TEST_SUITE_END()