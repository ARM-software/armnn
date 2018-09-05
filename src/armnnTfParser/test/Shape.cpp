//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct ShapeFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    ShapeFixture()
    {
        m_Prototext =
            "node { \n"
            "  name: \"Placeholder\" \n"
            "  op: \"Placeholder\" \n"
            "  attr { \n"
            "    key: \"dtype\" \n"
            "    value { \n"
            "      type: DT_FLOAT \n"
            "    } \n"
            "  } \n"
            "  attr { \n"
            "    key: \"shape\" \n"
            "    value { \n"
            "      shape { \n"
            "        dim { \n"
            "          size: 1 \n"
            "        } \n"
            "        dim { \n"
            "          size: 1 \n"
            "        } \n"
            "        dim { \n"
            "          size: 1 \n"
            "        } \n"
            "        dim { \n"
            "          size: 4 \n"
            "        } \n"
            "      } \n"
            "    } \n"
            "  } \n"
            "} \n"
            "node { \n"
            "  name: \"shapeTest\" \n"
            "  op: \"Shape\" \n"
            "  input: \"Placeholder\" \n"
            "  attr { \n"
            "    key: \"T\" \n"
            "    value { \n"
            "      type: DT_FLOAT \n"
            "    } \n"
            "  } \n"
            "  attr { \n"
            "    key: \"out_type\" \n"
            "    value { \n"
            "      type: DT_INT32 \n"
            "    } \n"
            "  } \n"
            "} \n"
            "node { \n"
            "  name: \"Reshape\" \n"
            "  op: \"Reshape\" \n"
            "  input: \"Placeholder\" \n"
            "  input: \"shapeTest\" \n"
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

        SetupSingleInputSingleOutput({1, 4}, "Placeholder", "Reshape");
    }
};

BOOST_FIXTURE_TEST_CASE(ParseShape, ShapeFixture)
{
    // Note: the test's output cannot be an int32 const layer, because ARMNN only supports u8 and float layers.
    //       For that reason I added a reshape layer which reshapes the input to its original dimensions.
    RunTest<2>({ 0.0f, 1.0f, 2.0f, 3.0f }, { 0.0f, 1.0f, 2.0f, 3.0f });
}

BOOST_AUTO_TEST_SUITE_END()
