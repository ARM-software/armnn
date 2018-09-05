//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

// Graph which tests that nodes are re-ordered in the queue when they are encountered a second time.
// In this case R0 will be encountered first via R1 and then via R2. At that time
// we need to make sure that R0 (and the I on which it is dependent) is moved to the front again
// so that it is before both R1 and R2.
//    I
//    |
//    R0
//   / \'
//  R1  R2
//   \  |
//    \ R3
//     \|
//      O
struct RediscoveredDependenciesFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    RediscoveredDependenciesFixture()
    {
        // Input = tf.placeholder(tf.float32, 1, "input")
        // Relu0 = tf.nn.relu(input, "relu0")
        // Relu1 = tf.nn.relu(relu0, "relu1")
        // Relu2 = tf.nn.relu(relu0, "relu2")
        // Relu3 = tf.nn.relu(relu2, "relu3")
        // Output = tf.add(relu1, relu3, "output")
        m_Prototext = R"(
            node {
              name: "input"
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
                  }
                }
              }
            }
            node {
              name: "relu0"
              op: "Relu"
              input: "input"
              attr {
                key: "T"
                value {
                  type: DT_FLOAT
                }
              }
            }
            node {
              name: "relu1"
              op: "Relu"
              input: "relu0"
              attr {
                key: "T"
                value {
                  type: DT_FLOAT
                }
              }
            }
            node {
              name: "relu2"
              op: "Relu"
              input: "relu0"
              attr {
                key: "T"
                value {
                  type: DT_FLOAT
                }
              }
            }
            node {
              name: "relu3"
              op: "Relu"
              input: "relu2"
              attr {
                key: "T"
                value {
                  type: DT_FLOAT
                }
              }
            }
            node {
              name: "output"
              op: "Add"
              input: "relu1"
              input: "relu3"
              attr {
                key: "T"
                value {
                  type: DT_FLOAT
                }
              }
            }
        )";
        SetupSingleInputSingleOutput({ 1 }, "input", "output");
    }
};

BOOST_FIXTURE_TEST_CASE(RediscoveredDependencies, RediscoveredDependenciesFixture)
{
    RunTest<1>({1}, {2});
}

// Tests that a simple cycle in the tensorflow graph will be detected and an exception thrown, rather than the TfParser
// getting stuck in an infinite loop.
BOOST_AUTO_TEST_CASE(SimpleCycle)
{
    const char* prototext = R"(
node {
  name: "r1"
  op: "Relu"
  input: "r2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "r2"
  op: "Relu"
  input: "r1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
    )";
    armnnTfParser::ITfParserPtr parser = armnnTfParser::ITfParser::Create();
    BOOST_CHECK_THROW(parser->CreateNetworkFromString(prototext, {}, { "r2" }), armnn::ParseException);
}

// Similar to the above SimpleCycle test, but has a single node which connects to itself.
BOOST_AUTO_TEST_CASE(SingleNodeCycle)
{
    const char* prototext = R"(
node {
  name: "r1"
  op: "Relu"
  input: "r1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
    )";
    armnnTfParser::ITfParserPtr parser = armnnTfParser::ITfParser::Create();
    BOOST_CHECK_THROW(parser->CreateNetworkFromString(prototext, {}, { "r1" }), armnn::ParseException);
}

// Similar to the above SimpleCycle test, but with a more complicated graph.
//    I
//    |
//    A2---<---<-
//   / \'        |
//  R1  R2       |
//   \  |        |
//    \ R3       |
//     \|        |
//      A1-->--->|
//
BOOST_AUTO_TEST_CASE(ComplexCycle)
{
    // Input = tf.placeholder(tf.float32, 1, "input")
    // Add2 = tf.nn.relu(input, add1, "add2") // This line won't actually run in TF, because add1 is not yet defined
    // Relu1 = tf.nn.relu(relu0, "relu1")
    // Relu2 = tf.nn.relu(relu0, "relu2")
    // Relu3 = tf.nn.relu(relu2, "relu3")
    // Add1 = tf.add(relu1, relu3, "add1")
    const char* prototext = R"(
        node {
            name: "input"
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
                }
            }
            }
        }
        node {
            name: "add2"
            op: "Add"
            input: "input"
            input: "add1"
            attr {
            key: "T"
            value {
                type: DT_FLOAT
            }
            }
        }
        node {
            name: "relu1"
            op: "Relu"
            input: "add2"
            attr {
            key: "T"
            value {
                type: DT_FLOAT
            }
            }
        }
        node {
            name: "relu2"
            op: "Relu"
            input: "add2"
            attr {
            key: "T"
            value {
                type: DT_FLOAT
            }
            }
        }
        node {
            name: "relu3"
            op: "Relu"
            input: "relu2"
            attr {
            key: "T"
            value {
                type: DT_FLOAT
            }
            }
        }
        node {
            name: "add1"
            op: "Add"
            input: "relu1"
            input: "relu3"
            attr {
            key: "T"
            value {
                type: DT_FLOAT
            }
            }
        }
    )";
    armnnTfParser::ITfParserPtr parser = armnnTfParser::ITfParser::Create();
    BOOST_CHECK_THROW(parser->CreateNetworkFromString(prototext, {}, { "add1" }), armnn::ParseException);
}

// Tests that a graph with an input that is not present throws a ParseException.
BOOST_AUTO_TEST_CASE(InvalidInput)
{
    const char* prototext = R"(
node {
  name: "r1"
  op: "Relu"
  input: "a-node-that-does-not-exist"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
    )";
    armnnTfParser::ITfParserPtr parser = armnnTfParser::ITfParser::Create();
    BOOST_CHECK_THROW(parser->CreateNetworkFromString(prototext, {}, { "r1" }), armnn::ParseException);
}

BOOST_AUTO_TEST_SUITE_END()
