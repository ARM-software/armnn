//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct UnsupportedMaximumFixture
    : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    UnsupportedMaximumFixture()
    {
        m_Prototext = R"(
            node {
                name: "graphInput"
                op: "Placeholder"
                attr {
                    key: "dtype"
                    value {
                        type: DT_FLOAT
                    }
                }
                attr {
                    key: "shape"
                    value {
                        shape {
                        }
                    }
                }
            }
            node {
                name: "Maximum"
                op: "Maximum"
                input: "graphInput"
                attr {
                    key: "dtype"
                    value {
                        type: DT_FLOAT
                    }
                }
            }
        )";
    }
};

BOOST_FIXTURE_TEST_CASE(UnsupportedMaximum, UnsupportedMaximumFixture)
{
    BOOST_CHECK_THROW(
        SetupSingleInputSingleOutput({ 1, 1 }, "graphInput", "Maximum"),
        armnn::ParseException);
}

struct SupportedMaximumFixture
    : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    SupportedMaximumFixture(const std::string & maxInput0,
                            const std::string & maxInput1,
                            const std::string & mulInput0,
                            const std::string & mulInput1)
    {
        m_Prototext = R"(
            node {
                name: "graphInput"
                op: "Placeholder"
                attr {
                    key: "dtype"
                    value { type: DT_FLOAT }
                }
                attr {
                    key: "shape"
                    value { shape { } }
                }
            }
            node {
                name: "Alpha"
                op: "Const"
                attr {
                    key: "dtype"
                    value { type: DT_FLOAT }
                }
                attr {
                    key: "value"
                    value {
                        tensor {
                            dtype: DT_FLOAT
                            tensor_shape {
                                dim { size: 1 }
                            }
                            float_val: 0.1
                        }
                    }
                }
            }
            node {
                name: "Mul"
                op: "Mul"
                input: ")" + mulInput0 + R"("
                input: ")" + mulInput1 + R"("
                attr {
                    key: "T"
                    value { type: DT_FLOAT }
                }
            }
            node {
                name: "Maximum"
                op: "Maximum"
                input: ")" + maxInput0 + R"("
                input: ")" + maxInput1 + R"("
                attr {
                    key: "T"
                    value { type: DT_FLOAT }
                }
            }
        )";
        SetupSingleInputSingleOutput({ 1, 2 }, "graphInput", "Maximum");
    }
};

struct LeakyRelu_Max_MulAT_T_Fixture : public SupportedMaximumFixture
{
    LeakyRelu_Max_MulAT_T_Fixture()
    : SupportedMaximumFixture("Mul","graphInput","Alpha","graphInput") {}
};

BOOST_FIXTURE_TEST_CASE(LeakyRelu_Max_MulAT_T, LeakyRelu_Max_MulAT_T_Fixture)
{
    RunTest<2>(std::vector<float>({-5.0, 3.0}), {-0.5, 3.0});
}

struct LeakyRelu_Max_T_MulAT_Fixture : public SupportedMaximumFixture
{
    LeakyRelu_Max_T_MulAT_Fixture()
    : SupportedMaximumFixture("graphInput","Mul","Alpha","graphInput") {}
};


BOOST_FIXTURE_TEST_CASE(LeakyRelu_Max_T_MulAT, LeakyRelu_Max_T_MulAT_Fixture)
{
    RunTest<2>(std::vector<float>({-10.0, 3.0}), {-1.0, 3.0});
}

struct LeakyRelu_Max_MulTA_T_Fixture : public SupportedMaximumFixture
{
    LeakyRelu_Max_MulTA_T_Fixture()
    : SupportedMaximumFixture("Mul", "graphInput","graphInput","Alpha") {}
};

BOOST_FIXTURE_TEST_CASE(LeakyRelu_Max_MulTA_T, LeakyRelu_Max_MulTA_T_Fixture)
{
    RunTest<2>(std::vector<float>({-5.0, 3.0}), {-0.5, 3.0});
}

struct LeakyRelu_Max_T_MulTA_Fixture : public SupportedMaximumFixture
{
    LeakyRelu_Max_T_MulTA_Fixture()
    : SupportedMaximumFixture("graphInput", "Mul", "graphInput", "Alpha") {}
};

BOOST_FIXTURE_TEST_CASE(LeakyRelu_Max_T_MulTA, LeakyRelu_Max_T_MulTA_Fixture)
{
    RunTest<2>(std::vector<float>({-10.0, 13.0}), {-1.0, 13.0});
}

BOOST_AUTO_TEST_SUITE_END()
