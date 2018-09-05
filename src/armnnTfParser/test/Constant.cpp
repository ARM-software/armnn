//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>

#include "armnnTfParser/ITfParser.hpp"

#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

// Tests that a Const node in Tensorflow can be converted to a ConstLayer in armnn (as opposed to most
// Const nodes which are used as weight inputs for convolutions etc. and are therefore not converted to
// armnn ConstLayers).
struct ConstantFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    ConstantFixture()
    {
        // Input = tf.placeholder(tf.float32, name = "input")
        // Const = tf.constant([17], tf.float32, [1])
        // Output = tf.add(input, const, name = "output")
        m_Prototext =
            R"(
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
        unknown_rank: true
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
        }
        float_val: 17.0
      }
    }
  }
}
node {
  name: "output"
  op: "Add"
  input: "input"
  input: "Const"
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

BOOST_FIXTURE_TEST_CASE(Constant, ConstantFixture)
{
    RunTest<1>({1}, {18});
}


// Tests that a single Const node in Tensorflow can be used twice by a dependant node. This should result in only
// a single armnn ConstLayer being created.
struct ConstantReusedFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    ConstantReusedFixture()
    {
        // Const = tf.constant([17], tf.float32, [1])
        // Output = tf.add(const, const, name = "output")
        m_Prototext =
            R"(
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
        }
        float_val: 17.0
      }
    }
  }
}
node {
  name: "output"
  op: "Add"
  input: "Const"
  input: "Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
            )";
        Setup({}, { "output" });
    }
};

BOOST_FIXTURE_TEST_CASE(ConstantReused, ConstantReusedFixture)
{
    RunTest<1>({}, { { "output", { 34 } } });
}

template <int ListSize>
struct ConstantValueListFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    ConstantValueListFixture()
    {
        m_Prototext =
            R"(
node {
  name: "output"
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
            size: 2
          }
          dim {
            size: 3
          }
        })";

        double value = 0.75;
        for (int i = 0; i < ListSize; i++, value += 0.25)
        {
            m_Prototext += std::string("float_val : ") + std::to_string(value) + "\n";
        }

        m_Prototext +=
            R"(
      }
    }
  }
}
            )";
        Setup({}, { "output" });
    }
};

using ConstantSingleValueListFixture = ConstantValueListFixture<1>;
using ConstantMultipleValueListFixture = ConstantValueListFixture<4>;
using ConstantMaxValueListFixture = ConstantValueListFixture<6>;

BOOST_FIXTURE_TEST_CASE(ConstantSingleValueList, ConstantSingleValueListFixture)
{
    RunTest<2>({}, { { "output", { 0.75f, 0.75f, 0.75f, 0.75f, 0.75f, 0.75f } } });
}
BOOST_FIXTURE_TEST_CASE(ConstantMultipleValueList, ConstantMultipleValueListFixture)
{
    RunTest<2>({}, { { "output", { 0.75f, 1.f, 1.25f, 1.5f,  1.5f,  1.5f } } });
}
BOOST_FIXTURE_TEST_CASE(ConstantMaxValueList, ConstantMaxValueListFixture)
{
    RunTest<2>({}, { { "output", { 0.75f, 1.f, 1.25f, 1.50f, 1.75f, 2.f } } });
}

template <bool WithShape, bool WithContent, bool WithValueList>
struct ConstantCreateFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    ConstantCreateFixture()
    {
        m_Prototext =
            R"(
node {
    name: "output"
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
            )";

        if (WithShape)
        {
            m_Prototext +=
                R"(
tensor_shape {
    dim {
    size: 2
    }
    dim {
    size: 2
    }
}
                )";
        }
        else
        {
            m_Prototext +=
                R"(
tensor_shape {
}
                )";
        }

        if (WithContent)
        {
            m_Prototext +=
                R"(
tensor_content: "\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?"
                )";
        }

        if (WithValueList)
        {
            m_Prototext +=
                R"(
float_val: 1.0
float_val: 1.0
float_val: 1.0
float_val: 1.0
float_val: 1.0
                )";
        }

        m_Prototext +=
            R"(
            }
        }
    }
}
            )";
    }
};

using ConstantCreateNoValueListFixture = ConstantCreateFixture<true, false, true>;
using ConstantCreateNoValueList2Fixture = ConstantCreateFixture<true, false, false>;
using ConstantCreateNoContentFixture = ConstantCreateFixture<true, true, false>;
using ConstantCreateNoContent2Fixture = ConstantCreateFixture<true, false, false>;
using ConstantCreateNoShapeFixture = ConstantCreateFixture<false, false, false>;
using ConstantCreateNoShape2Fixture = ConstantCreateFixture<false, true, false>;
using ConstantCreateNoShape3Fixture = ConstantCreateFixture<false, false, true>;

BOOST_FIXTURE_TEST_CASE(ConstantCreateInvalidValueList, ConstantCreateNoValueListFixture)
{
    BOOST_REQUIRE_THROW(Setup({}, { "output" }), armnn::ParseException);
}
BOOST_FIXTURE_TEST_CASE(ConstantCreateInvalidValueList2, ConstantCreateNoValueList2Fixture)
{
    BOOST_REQUIRE_THROW(Setup({}, { "output" }), armnn::ParseException);
}
BOOST_FIXTURE_TEST_CASE(ConstantCreateInvalidContent, ConstantCreateNoContentFixture)
{
    BOOST_REQUIRE_THROW(Setup({}, { "output" }), armnn::ParseException);
}
BOOST_FIXTURE_TEST_CASE(ConstantCreateInvalidShape, ConstantCreateNoShapeFixture)
{
    BOOST_REQUIRE_THROW(Setup({}, { "output" }), armnn::ParseException);
}
BOOST_FIXTURE_TEST_CASE(ConstantCreateNoShape2, ConstantCreateNoShape2Fixture)
{
    BOOST_REQUIRE_THROW(Setup({}, { "output" }), armnn::ParseException);
}
BOOST_FIXTURE_TEST_CASE(ConstantCreateNoShape3, ConstantCreateNoShape3Fixture)
{
    Setup({}, { "output" });
    RunTest<1>({}, { { "output", { 1.f, 1.f, 1.f, 1.f, 1.f } } });
}

BOOST_AUTO_TEST_SUITE_END()
