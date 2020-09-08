//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct ExpandDimsFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    ExpandDimsFixture(const std::string& expandDim)
    {
        m_Prototext =
                "node { \n"
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
                "  name: \"ExpandDims\" \n"
                "  op: \"ExpandDims\" \n"
                "  input: \"graphInput\" \n"
                "  attr { \n"
                "    key: \"T\" \n"
                "    value { \n"
                "      type: DT_FLOAT \n"
                "    } \n"
                "  } \n"
                "  attr { \n"
                "    key: \"Tdim\" \n"
                "    value { \n";
            m_Prototext += "i:" + expandDim;
            m_Prototext +=
                "    } \n"
                "  } \n"
                "} \n";

        SetupSingleInputSingleOutput({ 2, 3, 5 }, "graphInput", "ExpandDims");
    }
};

struct ExpandZeroDim : ExpandDimsFixture
{
    ExpandZeroDim() : ExpandDimsFixture("0") {}
};

struct ExpandTwoDim : ExpandDimsFixture
{
    ExpandTwoDim() : ExpandDimsFixture("2") {}
};

struct ExpandThreeDim : ExpandDimsFixture
{
    ExpandThreeDim() : ExpandDimsFixture("3") {}
};

struct ExpandMinusOneDim : ExpandDimsFixture
{
    ExpandMinusOneDim() : ExpandDimsFixture("-1") {}
};

struct ExpandMinusThreeDim : ExpandDimsFixture
{
    ExpandMinusThreeDim() : ExpandDimsFixture("-3") {}
};

BOOST_FIXTURE_TEST_CASE(ParseExpandZeroDim, ExpandZeroDim)
{
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo("ExpandDims").second.GetShape() ==
                armnn::TensorShape({1, 2, 3, 5})));
}

BOOST_FIXTURE_TEST_CASE(ParseExpandTwoDim, ExpandTwoDim)
{
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo("ExpandDims").second.GetShape() ==
                armnn::TensorShape({2, 3, 1, 5})));
}

BOOST_FIXTURE_TEST_CASE(ParseExpandThreeDim, ExpandThreeDim)
{
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo("ExpandDims").second.GetShape() ==
                armnn::TensorShape({2, 3, 5, 1})));
}

BOOST_FIXTURE_TEST_CASE(ParseExpandMinusOneDim, ExpandMinusOneDim)
{
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo("ExpandDims").second.GetShape() ==
                armnn::TensorShape({2, 3, 5, 1})));
}

BOOST_FIXTURE_TEST_CASE(ParseExpandMinusThreeDim, ExpandMinusThreeDim)
{
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo("ExpandDims").second.GetShape() ==
                armnn::TensorShape({2, 1, 3, 5})));
}

struct ExpandDimsAsInputFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    ExpandDimsAsInputFixture(const std::string& expandDim,
                             const bool wrongDataType = false,
                             const std::string& numElements = "1")
    {
        std::string dataType = (wrongDataType) ? "DT_FLOAT" : "DT_INT32";
        std::string val = (wrongDataType) ? ("float_val: " + expandDim + ".0") : ("int_val: "+ expandDim);

        m_Prototext = R"(
        node {
            name: "a"
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
                        dim {
                            size: 1
                        }
                        dim {
                            size: 4
                        }
                    }
                }
            }
        }
        node {
            name: "b"
            op: "Const"
            attr {
                key: "dtype"
                value {
                    type:  )" + dataType + R"(
                }
            }
            attr {
                key: "value"
                value {
                    tensor {
                        dtype: )" + dataType + R"(
                        tensor_shape {
                            dim {
                                size: )" + numElements + R"(
                            }
                        }
                        )" + val + R"(
                    }
                }
            }
        }
        node {
            name: "ExpandDims"
            op: "ExpandDims"
            input: "a"
            input: "b"
            attr {
                key: "T"
                value {
                    type: DT_FLOAT
                }
            }
            attr {
                key: "Tdim"
                value {
                    type: DT_INT32
                }
            }
        }
        versions {
            producer: 134
        })";
    }
};

struct ExpandDimAsInput : ExpandDimsAsInputFixture
{
    ExpandDimAsInput() : ExpandDimsAsInputFixture("0")
    {
        Setup({{"a", {1,4}} ,{"b",{1,1}}}, { "ExpandDims" });
    }
};


BOOST_FIXTURE_TEST_CASE(ParseExpandDimAsInput, ExpandDimAsInput)
{
    // Axis parameter that describes which axis/dim should be expanded is passed as a second input
    BOOST_TEST((m_Parser->GetNetworkOutputBindingInfo("ExpandDims").second.GetShape() ==
                armnn::TensorShape({1, 1, 4})));
}

struct ExpandDimAsInputWrongDataType : ExpandDimsAsInputFixture
{
    ExpandDimAsInputWrongDataType() : ExpandDimsAsInputFixture("0", true, "1") {}
};

BOOST_FIXTURE_TEST_CASE(ParseExpandDimAsInputWrongDataType, ExpandDimAsInputWrongDataType)
{
    // Axis parameter that describes which axis/dim should be expanded is passed as a second input
    // Axis parameter is of wrong data type (float instead of int32)
    BOOST_REQUIRE_THROW(Setup({{"a", {1,4}} ,{"b",{1,1}}}, { "ExpandDims" }), armnn::ParseException);
}

struct ExpandDimAsInputWrongShape : ExpandDimsAsInputFixture
{
    ExpandDimAsInputWrongShape() : ExpandDimsAsInputFixture("0", false, "2") {}
};

BOOST_FIXTURE_TEST_CASE(ParseExpandDimAsInputWrongShape, ExpandDimAsInputWrongShape)
{
    // Axis parameter that describes which axis/dim should be expanded is passed as a second input
    // Axis parameter is of wrong shape
    BOOST_REQUIRE_THROW(Setup({{"a", {1,4}} ,{"b",{1,1}}}, { "ExpandDims" }), armnn::ParseException);
}

struct ExpandDimsAsNotConstInputFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    ExpandDimsAsNotConstInputFixture()
    {
        m_Prototext = R"(
            node {
                name: "a"
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
                            dim {
                                size: 1
                            }
                            dim {
                            size: 4
                            }
                        }
                    }
                }
            }
            node {
                name: "b"
                op: "Placeholder"
                attr {
                    key: "dtype"
                        value {
                            type: DT_INT32
                        }
                }
                attr {
                    key: "shape"
                    value {
                        shape {
                            dim {
                                size: 1
                            }
                        }
                    }
                }
            }
            node {
                name: "ExpandDims"
                op: "ExpandDims"
                input: "a"
                input: "b"
                attr {
                    key: "T"
                        value {
                            type: DT_FLOAT
                        }
                    }
                    attr {
                        key: "Tdim"
                        value {
                            type: DT_INT32
                        }
                    }
                }
            versions {
                producer: 134
            })";
    }
};

BOOST_FIXTURE_TEST_CASE(ParseExpandDimAsNotConstInput, ExpandDimsAsNotConstInputFixture)
{
    // Axis parameter that describes which axis/dim should be expanded is passed as a second input.
    // But is not a constant tensor --> not supported
    BOOST_REQUIRE_THROW(Setup({{"a", {1,4}} ,{"b",{1,1}}}, { "ExpandDims" }),
                        armnn::ParseException);
}

BOOST_AUTO_TEST_SUITE_END()
