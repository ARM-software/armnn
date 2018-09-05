//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"
#include "Runtime.hpp"
#include "Network.hpp"
#include "Graph.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

// In Tensorflow fully connected layers are expressed as a MatMul followed by an Add.
// The TfParser must detect this case and convert them to a FullyConnected layer.
struct FullyConnectedFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    FullyConnectedFixture()
    {
        // Input = tf.placeholder(tf.float32, [1, 1], "input")
        // Weights = tf.constant([2], tf.float32, [1, 1])
        // Matmul = tf.matmul(input, weights)
        // Bias = tf.constant([1], tf.float32)
        // Output = tf.add(matmul, bias, name="output")
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
        dim {
          size: 1
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
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
          dim {
            size: 1
          }
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "MatMul"
  op: "MatMul"
  input: "input"
  input: "Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
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
          dim {
            size: 1
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "output"
  op: "Add"
  input: "MatMul"
  input: "Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
        )";
        SetupSingleInputSingleOutput({ 1, 1 }, "input", "output");
    }
};

BOOST_FIXTURE_TEST_CASE(FullyConnected, FullyConnectedFixture)
{
    RunTest<1>({ 3 }, { 7 });
}

// Similar to FullyConnectedFixture, but this time the MatMul's output is used by two Adds. This should result
// in two FullyConnected layers being created.
//      I
//      |
//      M -- C
//     / \'
// C-- A  A -- C
//     \ /
//      A
struct MatMulUsedInTwoFcFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    MatMulUsedInTwoFcFixture()
    {
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
        dim {
          size: 1
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
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
          dim {
            size: 1
          }
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "MatMul"
  op: "MatMul"
  input: "input"
  input: "Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
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
          dim {
            size: 1
          }
        }
        float_val: 5.0
      }
    }
  }
}
node {
  name: "Const_2"
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
          dim {
            size: 1
          }
        }
        float_val: 15.0
      }
    }
  }
}
node {
  name: "Add"
  op: "Add"
  input: "MatMul"
  input: "Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Add_1"
  op: "Add"
  input: "MatMul"
  input: "Const_2"
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
  input: "Add"
  input: "Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
        )";
        SetupSingleInputSingleOutput({ 1, 1 }, "input", "output");
    }
};

BOOST_FIXTURE_TEST_CASE(MatMulUsedInTwoFc, MatMulUsedInTwoFcFixture)
{
    RunTest<1>({ 3 }, { 32 });
    // Ideally we would check here that the armnn network has 5 layers:
    //  Input, 2 x FullyConnected (biased), Add and Output.
    // This would make sure the parser hasn't incorrectly added some unconnected layers corresponding to the MatMul.
}

// Similar to MatMulUsedInTwoFc, but this time the Adds are 'staggered' (see diagram), which means that only one
// FullyConnected layer can be created (the other should just be an Add).
//        I
//        |
//        M -- C1
//       / \'
// C2 -- A  |
//       \ /
//        A
struct MatMulUsedInTwoFcStaggeredFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    MatMulUsedInTwoFcStaggeredFixture()
    {
        // Input = tf.placeholder(tf.float32, shape=[1,1], name = "input")
        // Const1 = tf.constant([17], tf.float32, [1,1])
        // Mul = tf.matmul(input, const1)
        // Monst2 = tf.constant([7], tf.float32, [1])
        // Fc = tf.add(mul, const2)
        // Output = tf.add(mul, fc, name="output")
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
        dim {
          size: 1
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
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
          dim {
            size: 1
          }
        }
        float_val: 17.0
      }
    }
  }
}
node {
  name: "MatMul"
  op: "MatMul"
  input: "input"
  input: "Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
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
          dim {
            size: 1
          }
        }
        float_val: 7.0
      }
    }
  }
}
node {
  name: "Add"
  op: "Add"
  input: "MatMul"
  input: "Const_1"
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
  input: "MatMul"
  input: "Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
        )";
        SetupSingleInputSingleOutput({ 1, 1 }, "input", "output");
    }
};

BOOST_FIXTURE_TEST_CASE(MatMulUsedInTwoFcStaggered, MatMulUsedInTwoFcStaggeredFixture)
{
    RunTest<1>({ 2 }, { 75 });
    // Ideally we would check here that the armnn network has 5 layers:
    //   Input, FullyConnected (biased), FullyConnected (non biased), Add and Output.
}

// A MatMul in isolation, not connected to an add. Should result in a non-biased FullyConnected layer.
struct MatMulFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    MatMulFixture()
    {
        // Input = tf.placeholder(tf.float32, shape = [1, 1], name = "input")
        // Const = tf.constant([17], tf.float32, [1, 1])
        //  Output = tf.matmul(input, const, name = "output")
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
        dim {
          size: 1
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
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
          dim {
            size: 1
          }
        }
        float_val: 17.0
      }
    }
  }
}
node {
  name: "output"
  op: "MatMul"
  input: "input"
  input: "Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
        )";
        SetupSingleInputSingleOutput({ 1, 1 }, "input", "output");
    }
};

BOOST_FIXTURE_TEST_CASE(MatMul, MatMulFixture)
{
    RunTest<1>({ 2 }, { 34 });
}

BOOST_AUTO_TEST_SUITE_END()
