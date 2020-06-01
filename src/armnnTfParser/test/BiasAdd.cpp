//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct BiasAddFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    explicit BiasAddFixture(const std::string& dataFormat)
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
  name: "bias"
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
            size: 3
          }
        }
        float_val: 1
        float_val: 2
        float_val: 3
      }
    }
  }
}
node {
  name: "biasAdd"
  op : "BiasAdd"
  input: "graphInput"
  input: "bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: ")" + dataFormat + R"("
    }
  }
}
)";

        SetupSingleInputSingleOutput({ 1, 3, 1, 3 }, "graphInput", "biasAdd");
    }
};

struct BiasAddFixtureNCHW : BiasAddFixture
{
    BiasAddFixtureNCHW() : BiasAddFixture("NCHW") {}
};

struct BiasAddFixtureNHWC : BiasAddFixture
{
    BiasAddFixtureNHWC() : BiasAddFixture("NHWC") {}
};

BOOST_FIXTURE_TEST_CASE(ParseBiasAddNCHW, BiasAddFixtureNCHW)
{
    RunTest<4>(std::vector<float>(9), { 1, 1, 1, 2, 2, 2, 3, 3, 3 });
}

BOOST_FIXTURE_TEST_CASE(ParseBiasAddNHWC, BiasAddFixtureNHWC)
{
    RunTest<4>(std::vector<float>(9), { 1, 2, 3, 1, 2, 3, 1, 2, 3 });
}

BOOST_AUTO_TEST_SUITE_END()
