//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct ReshapeFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    ReshapeFixture()
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
            "node { \n"
            "  name: \"Reshape/shape\" \n"
            "  op: \"Const\" \n"
            "  attr { \n"
            "    key: \"dtype\" \n"
            "    value { \n"
            "      type: DT_INT32 \n"
            "    } \n"
            "  } \n"
            "  attr { \n"
            "    key: \"value\" \n"
            "    value { \n"
            "      tensor { \n"
            "        dtype: DT_INT32 \n"
            "        tensor_shape { \n"
            "          dim { \n"
            "            size: 2 \n"
            "          } \n"
            "        } \n"
            "        tensor_content: \"\\002\\000\\000\\000\\002\\000\\000\\000\" \n"
            "      } \n"
            "    } \n"
            "  } \n"
            "} \n"
            "node { \n"
            "  name: \"Reshape\" \n"
            "  op: \"Reshape\" \n"
            "  input: \"graphInput\" \n"
            "  input: \"Reshape/shape\" \n"
            "  attr { \n"
            "    key: \"T\" \n"
            "    value { \n"
            "      type: DT_FLOAT \n"
            "    } \n"
            "  } \n"
            "  attr { \n"
            "    key: \"Tshape\" \n"
            "    value { \n"
            "      type: DT_INT32 \n"
            "    } \n"
            "  } \n"
            "} \n";

        SetupSingleInputSingleOutput({1, 4}, "graphInput", "Reshape");
    }
};

BOOST_FIXTURE_TEST_CASE(ParseReshape, ReshapeFixture)
{
    RunTest<2>({ 0.0f, 1.0f, 2.0f, 3.0f }, { 0.0f, 1.0f, 2.0f, 3.0f });
}

BOOST_AUTO_TEST_SUITE_END()
