//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

#include <array>

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct FusedBatchNormFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    explicit FusedBatchNormFixture(const std::string& dataLayout)
    {
        m_Prototext = "node { \n"
            "  name: \"graphInput\" \n"
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
            "      } \n"
            "    } \n"
            "  } \n"
            "} \n"
            "node { \n"
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
            "        } \n"
            "        float_val: 1.0 \n"
            "      } \n"
            "    } \n"
            "  } \n"
            "} \n"
            "node { \n"
            "  name: \"Const_2\" \n"
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
            "        } \n"
            "        float_val: 0.0 \n"
            "      } \n"
            "    } \n"
            "  } \n"
            "} \n"
            "node { \n"
            "  name: \"FusedBatchNormLayer/mean\" \n"
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
            "        } \n"
            "        float_val: 5.0 \n"
            "      } \n"
            "    } \n"
            "  } \n"
            "} \n"
            "node { \n"
            "  name: \"FusedBatchNormLayer/variance\" \n"
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
            "        } \n"
            "        float_val: 2.0 \n"
            "      } \n"
            "    } \n"
            "  } \n"
            "} \n"
            "node { \n"
            "  name: \"output\" \n"
            "  op: \"FusedBatchNorm\" \n"
            "  input: \"graphInput\" \n"
            "  input: \"Const_1\" \n"
            "  input: \"Const_2\" \n"
            "  input: \"FusedBatchNormLayer/mean\" \n"
            "  input: \"FusedBatchNormLayer/variance\" \n"
            "  attr { \n"
            "    key: \"T\" \n"
            "    value { \n"
            "      type: DT_FLOAT \n"
            "    } \n"
            "  } \n";

        // NOTE: we only explicitly set data_format when it is not the default NHWC
        if (dataLayout != "NHWC")
        {
            m_Prototext.append("  attr { \n"
                "    key: \"data_format\" \n"
                "    value { \n"
                "      s: \"");
            m_Prototext.append(dataLayout);
            m_Prototext.append("\" \n"
                "    } \n"
                "  } \n");
        }

        m_Prototext.append("  attr { \n"
                           "    key: \"epsilon\" \n"
                           "    value { \n"
                           "      f: 0.0010000000475 \n"
                           "    } \n"
                           "  } \n"
                           "  attr { \n"
                           "    key: \"is_training\" \n"
                           "    value { \n"
                           "      b: false \n"
                           "    } \n"
                           "  } \n"
                           "} \n");

        // Set the input shape according to the data layout
        std::array<unsigned int, 4> dims;
        if (dataLayout == "NHWC")
        {
            dims = { 1u, 3u, 3u, 1u };
        }
        else // dataLayout == "NCHW"
        {
            dims = { 1u, 1u, 3u, 3u };
        }

        SetupSingleInputSingleOutput(armnn::TensorShape(4, dims.data()), "graphInput", "output");
    }
};

struct FusedBatchNormNhwcFixture : FusedBatchNormFixture
{
    FusedBatchNormNhwcFixture() : FusedBatchNormFixture("NHWC"){}
};
BOOST_FIXTURE_TEST_CASE(ParseFusedBatchNormNhwc, FusedBatchNormNhwcFixture)
{
    RunTest<4>({ 1, 2, 3, 4, 5, 6, 7, 8, 9 },               // Input data.
               { -2.8277204f, -2.12079024f, -1.4138602f,
                 -0.7069301f,  0.0f,         0.7069301f,
                  1.4138602f,  2.12079024f,  2.8277204f }); // Expected output data.
}

struct FusedBatchNormNchwFixture : FusedBatchNormFixture
{
    FusedBatchNormNchwFixture() : FusedBatchNormFixture("NCHW"){}
};
BOOST_FIXTURE_TEST_CASE(ParseFusedBatchNormNchw, FusedBatchNormNchwFixture)
{
    RunTest<4>({ 1, 2, 3, 4, 5, 6, 7, 8, 9 },               // Input data.
               { -2.8277204f, -2.12079024f, -1.4138602f,
                 -0.7069301f,  0.0f,         0.7069301f,
                  1.4138602f,  2.12079024f,  2.8277204f }); // Expected output data.
}

BOOST_AUTO_TEST_SUITE_END()
