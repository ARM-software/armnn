//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

#include <armnn/utility/IgnoreUnused.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct SplitFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    SplitFixture(bool withDimZero=false) {
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
          name: "graphInput2"
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
        name: "multiplication"
        op : "Mul"
        input: "graphInput"
        input: "graphInput2"
        attr {
        key: "T"
        value {
            type: DT_FLOAT
        }
        }
        }
        node {
          name: "SplitInput"
          op: "Const"
          attr {
            key: "dtype"
            value {
              type: DT_INT32
            }
          }
          attr {
            key: "value"
            value {
              tensor {
                dtype: DT_INT32
                tensor_shape {
                }
                int_val: )";

        if(withDimZero)
        {
            m_Prototext += std::to_string(3);
        }
        else
        {
            m_Prototext += std::to_string(1);
        }

        m_Prototext += R"(
        }
        }
        }
        }
        node {
          name: "Split"
          op: "Split" )";
        if(withDimZero)
        {
            m_Prototext += "input: \"SplitInput\"\n";
            m_Prototext += "input: \"multiplication\"\n";
        }
        else
        {
            m_Prototext += "input: \"graphInput\"\n";
            m_Prototext += "input: \"SplitInput\"\n";
        }
        m_Prototext += R"(
          attr {
            key: "num_split"
            value {
              i: 2
            }
          }
        }
        node {
            name: "Relu_1"
            op: "Relu"
            input: "Split:0"
            attr {
            key: "T"
            value {
            type: DT_FLOAT
             }
            }
            }
         node {
            name: "Relu_2"
            op: "Relu"
            input:"Split:1"
            attr {
            key: "T"
            value {
            type: DT_FLOAT
             }
            }
            } )";

        Setup( { { "graphInput", { 1,  2,  2 , 2} } , { "graphInput2", { 1,  2,  2 , 2} }},
               { "Relu_1", "Relu_2" });
    }
};

struct InputFirstSplitFixture : SplitFixture
{
    InputFirstSplitFixture() : SplitFixture(true) {}
};

BOOST_FIXTURE_TEST_CASE(ParseAxisOneSplitTwo, SplitFixture)
{
    BOOST_TEST(
        (m_Parser->GetNetworkOutputBindingInfo("Relu_1").second.GetShape() == armnn::TensorShape({ 1, 1, 2, 2 })));

    BOOST_TEST(
        (m_Parser->GetNetworkOutputBindingInfo("Relu_2").second.GetShape() == armnn::TensorShape({ 1, 1, 2, 2 })));

    RunTest<4>({ { "graphInput", { -1.0f, -0.5f, 1.25f, -3.0f, 0.0f, 0.5f, -0.75f, 1.75f } } },
               { { "Relu_1", { 0.0f, 0.0f, 1.25f, 0.0f } },
                 { "Relu_2", { 0.0f, 0.5f, 0.0f, 1.75f } } });
}

BOOST_FIXTURE_TEST_CASE(ParseSplit, InputFirstSplitFixture)
{

    BOOST_TEST(
            (m_Parser->GetNetworkOutputBindingInfo("Relu_1").second.GetShape() == armnn::TensorShape({ 1, 2, 2, 1 })));

    BOOST_TEST(
            (m_Parser->GetNetworkOutputBindingInfo("Relu_2").second.GetShape() == armnn::TensorShape({ 1, 2, 2, 1 })));

    RunTest<4>({ { "graphInput", { -1.0f, -0.5f, 1.25f, -3.0f, 0.0f, 0.5f, -0.75f , 1.75f } } ,
                 { "graphInput2", { -1.0f, -0.5f, 1.25f, -3.0f, 0.0f, 0.5f, -0.75f , 1.75f } } },
               { { "Relu_1", { 1.0f, 1.5625f, 0, 0.5625f } },
                 { "Relu_2", { 0.25, 9.0f, 0.25f, 3.0625f } } });
}

struct SplitLastDimFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    SplitLastDimFixture(bool withDimZero=false) {
        armnn::IgnoreUnused(withDimZero);
        m_Prototext = R"(
        node {
          name: "Placeholder"
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
                  size: 2
                }
                dim {
                  size: 2
                }
                dim {
                  size: 3
                }
              }
            }
          }
        }
        node {
          name: "Const"
          op: "Const"
          attr {
            key: "dtype"
            value {
              type: DT_INT32
            }
          }
          attr {
            key: "value"
            value {
              tensor {
                dtype: DT_INT32
                tensor_shape {
                }
                int_val: 3
              }
            }
          }
        }
        node {
          name: "split/split_dim"
          op: "Const"
          attr {
            key: "dtype"
            value {
              type: DT_INT32
            }
          }
          attr {
            key: "value"
            value {
              tensor {
                dtype: DT_INT32
                tensor_shape {
                }
                int_val: 3
              }
            }
          }
        }
        node {
          name: "split"
          op: "Split"
          input: "split/split_dim"
          input: "Placeholder"
          attr {
            key: "T"
            value {
              type: DT_FLOAT
            }
          }
          attr {
            key: "num_split"
            value {
              i: 3
            }
          }
        }
        node {
          name: "sub0/y"
          op: "Const"
          attr {
            key: "dtype"
            value {
              type: DT_FLOAT
            }
          }
          attr {
            key: "value"
            value {
              tensor {
                dtype: DT_FLOAT
                tensor_shape {
                }
                float_val: 3.0
              }
            }
          }
        }
        node {
          name: "sub0"
          op: "Sub"
          input: "split"
          input: "sub0/y"
          attr {
            key: "T"
            value {
              type: DT_FLOAT
            }
          }
        }
        node {
          name: "sub1/y"
          op: "Const"
          attr {
            key: "dtype"
            value {
              type: DT_FLOAT
            }
          }
          attr {
            key: "value"
            value {
              tensor {
                dtype: DT_FLOAT
                tensor_shape {
                }
                float_val: 2.0
              }
            }
          }
        }
        node {
          name: "sub1"
          op: "Sub"
          input: "split:1"
          input: "sub1/y"
          attr {
            key: "T"
            value {
              type: DT_FLOAT
            }
          }
        }
        node {
          name: "sub2/y"
          op: "Const"
          attr {
            key: "dtype"
            value {
              type: DT_FLOAT
            }
          }
          attr {
            key: "value"
            value {
              tensor {
                dtype: DT_FLOAT
                tensor_shape {
                }
                float_val: 1.0
              }
            }
          }
        }
        node {
          name: "sub2"
          op: "Sub"
          input: "split:2"
          input: "sub2/y"
          attr {
            key: "T"
            value {
              type: DT_FLOAT
            }
          }
        }
        versions {
          producer: 27
        } )";

        Setup( { { "Placeholder", { 1,  2,  2 , 3} } },
               { "sub0", "sub1", "sub2" });
    }
};

BOOST_FIXTURE_TEST_CASE(SplitLastDimTest, SplitLastDimFixture)
{
    BOOST_TEST(
            (m_Parser->GetNetworkOutputBindingInfo("sub0").second.GetShape() == armnn::TensorShape({ 1, 2, 2, 1 })));

    BOOST_TEST(
            (m_Parser->GetNetworkOutputBindingInfo("sub1").second.GetShape() == armnn::TensorShape({ 1, 2, 2, 1 })));

    BOOST_TEST(
            (m_Parser->GetNetworkOutputBindingInfo("sub2").second.GetShape() == armnn::TensorShape({ 1, 2, 2, 1 })));

    RunTest<4>({ { "Placeholder", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f } } },
               { { "sub0", { -2.0f, 1.0f, 4.0f, 7.0f } },
                 { "sub1", { 0.0f, 3.0f, 6.0f, 9.0f } },
                 { "sub2", { 2.0f, 5.0f, 8.0f, 11.0f } } });
}

BOOST_AUTO_TEST_SUITE_END()
