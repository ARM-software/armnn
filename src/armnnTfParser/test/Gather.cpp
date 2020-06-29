//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnTfParser/ITfParser.hpp"

#include "ParserPrototxtFixture.hpp"
#include <PrototxtConversions.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(TensorflowParser)

namespace {
// helper for setting the dimensions in prototxt
void dimsHelper(const std::vector<int>& dims, std::string& text){
    for(unsigned int i = 0; i < dims.size(); ++i) {
        text.append(R"(dim {
      size: )");
        text.append(std::to_string(dims[i]));
        text.append(R"(
    })");
    }
}

// helper for converting from integer to octal representation
void octalHelper(const std::vector<int>& indicesContent, std::string& text){
    for(unsigned int i = 0; i < indicesContent.size(); ++i) {
        text.append(armnnUtils::ConvertInt32ToOctalString(static_cast<int>(indicesContent[i])));
    }
}
} // namespace

struct GatherFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    GatherFixture(const armnn::TensorShape& inputShape0,
                  const armnn::TensorShape& inputShape1,
                  const std::vector<int>& input1Content,
                  const std::vector<int>& input0Dims,
                  const std::vector<int>& input1Dims,
                  int axis = 0)
    {
        m_Prototext = R"(
node {
  name: "input0"
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
)";
        dimsHelper(input0Dims, m_Prototext);

        m_Prototext.append(R"(
      }
    }
  }
}
node {
  name: "input1"
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
)");
        dimsHelper(input1Dims, m_Prototext);

        m_Prototext.append(R"(
        }
        tensor_content: ")");
        octalHelper(input1Content, m_Prototext);
        m_Prototext.append(R"("
      }
    }
  }
}
node {
  name: "output"
  op: "Gather"
  input: "input0"
  input: "input1"
  attr {
    key: "Tindices"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tparams"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "axis"
    value {
      i:  )");
        m_Prototext += std::to_string(axis);

        m_Prototext.append(R"(
    }
  }
}
        )");

        Setup({ { "input0", inputShape0 },
                { "input1", inputShape1 } },
              { "output" });

    }
};


struct GatherFixture1DParams1DIndices : public GatherFixture
{
    GatherFixture1DParams1DIndices() : GatherFixture(
            { 4, 1, 1, 1 },
            { 4, 0, 0, 0 },
            { 0, 2, 1, 3 },
            { 4 },
            { 4 },
            0) {}
};

struct GatherFixture1DParamsMultiDimIndices : public GatherFixture
{
    GatherFixture1DParamsMultiDimIndices() : GatherFixture(
            { 4, 1, 1 },
            { 2, 2, 1, 1 },
            { 0, 1, 1, 3 },
            { 4 },
            { 2, 2 },
            0) {}
};

struct GatherFixtureMultiDimParamMultiDimIndices : public GatherFixture
{
    GatherFixtureMultiDimParamMultiDimIndices() : GatherFixture(
            { 5, 2, 1 },
            { 2, 1, 4 },
            { 1, 3, 0, 2 },
            { 5, 2 },
            { 2, 2 },
            0) {}
};

BOOST_FIXTURE_TEST_CASE(ParseGather1DParams1DIndices, GatherFixture1DParams1DIndices)
{
    RunTest<4>({ { "input0", { 1, 2, 3, 4 } } },

               { { "output", { 1, 3, 2, 4 } } });
}

BOOST_FIXTURE_TEST_CASE(ParseGather1DParamsMultiDimIndices, GatherFixture1DParamsMultiDimIndices)
{
    RunTest<4>({ { "input0", { 1, 2, 3, 4 } } },

               { { "output", { 1, 2, 2, 4 } } });
}

BOOST_FIXTURE_TEST_CASE(ParseGatherMultiDimParamMultiDimIndices, GatherFixtureMultiDimParamMultiDimIndices)
{
    RunTest<4>({ { "input0", { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 } } },

               { { "output", { 3, 4, 7, 8, 1, 2, 5, 6} } });
}

BOOST_AUTO_TEST_SUITE_END()
