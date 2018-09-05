//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"
// This is a special case for add, which supports broadcasting.
BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct BroadcastForAddFixtureSlot1 : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    BroadcastForAddFixtureSlot1()
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
          name: "Const_1"
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
                float_val: 4.0
                float_val: 5.0
              }
            }
          }
        }
        node {
          name: "Add"
          op: "Add"
          input: "graphInput"
          input: "Const_1"
          attr {
            key: "T"
            value {
              type: DT_FLOAT
            }
          }
        }
        )";

        SetupSingleInputSingleOutput({ 1, 2, 2, 2 }, "graphInput", "Add");
    }
};

struct BroadcastForAddFixtureSlot0 : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    BroadcastForAddFixtureSlot0()
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
      name: "Const_1"
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
            float_val: 4.0
            float_val: 5.0
          }
        }
      }
    }
    node {
      name: "Add"
      op: "Add"
      input: "Const_1"
      input: "graphInput"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    )";

        SetupSingleInputSingleOutput({ 1, 2, 2, 2 }, "graphInput", "Add");
    }
};


BOOST_FIXTURE_TEST_CASE(ParseBroadcastForAddition1, BroadcastForAddFixtureSlot1)
{
    RunTest<4>({ 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0 }, { 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0 });
}

BOOST_FIXTURE_TEST_CASE(ParseBroadcastForAddition0, BroadcastForAddFixtureSlot0)
{
    RunTest<4>({ 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0 }, { 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0 });
}



BOOST_AUTO_TEST_SUITE_END()
