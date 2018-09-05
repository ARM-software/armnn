//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct MultiInputsOutputsFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    MultiInputsOutputsFixture()
    {
        // Input1 = tf.placeholder(tf.float32, shape=[], name = "input1")
        // Input2 = tf.placeholder(tf.float32, shape = [], name = "input2")
        // Add1 = tf.add(input1, input2, name = "add1")
        // Add2 = tf.add(input1, input2, name = "add2")
        m_Prototext = R"(
node {
  name: "input1"
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
  name: "input2"
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
  name: "add1"
  op: "Add"
  input: "input1"
  input: "input2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "add2"
  op: "Add"
  input: "input1"
  input: "input2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
        )";
        Setup({ { "input1", { 1 } },
                { "input2", { 1 } } },
              { "add1", "add2" });
    }
};

BOOST_FIXTURE_TEST_CASE(MultiInputsOutputs, MultiInputsOutputsFixture)
{
    RunTest<1>({ { "input1", {12.0f} }, { "input2", { 13.0f } } },
               { { "add1", { 25.0f } }, { "add2", { 25.0f } } });
}

BOOST_AUTO_TEST_SUITE_END()
