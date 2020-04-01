//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/utility/Assert.hpp>
#include <boost/test/unit_test.hpp>

#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

#include <map>
#include <string>


BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct AddNFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    AddNFixture(const std::vector<armnn::TensorShape> inputShapes, unsigned int numberOfInputs)
    {
        ARMNN_ASSERT(inputShapes.size() == numberOfInputs);
        m_Prototext = "";
        for (unsigned int i = 0; i < numberOfInputs; i++)
        {
            m_Prototext.append("node { \n");
            m_Prototext.append("  name: \"input").append(std::to_string(i)).append("\"\n");
            m_Prototext += R"(  op: "Placeholder"
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
)";
        }
        m_Prototext += R"(node {
  name:  "output"
  op: "AddN"
)";
        for (unsigned int i = 0; i < numberOfInputs; i++)
        {
            m_Prototext.append("  input: \"input").append(std::to_string(i)).append("\"\n");
        }
        m_Prototext += R"(  attr {
    key: "N"
    value {
)";
        m_Prototext.append("      i: ").append(std::to_string(numberOfInputs)).append("\n");
        m_Prototext += R"(    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
})";

        std::map<std::string, armnn::TensorShape> inputs;
        for (unsigned int i = 0; i < numberOfInputs; i++)
        {
            std::string name("input");
            name.append(std::to_string(i));
            inputs.emplace(std::make_pair(name, inputShapes[i]));
        }
        Setup(inputs, {"output"});
    }

};

// try with 2, 3, 5 and 8 inputs
struct FiveTwoDimInputsFixture : AddNFixture
{
    FiveTwoDimInputsFixture() : AddNFixture({ { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 } }, 5) {}
};


BOOST_FIXTURE_TEST_CASE(FiveTwoDimInputs, FiveTwoDimInputsFixture)
{
    RunTest<2>({ { "input0", { 1.0, 2.0, 3.0, 4.0 } },
                 { "input1", { 1.0, 5.0, 2.0, 2.0 } },
                 { "input2", { 1.0, 1.0, 2.0, 2.0 } },
                 { "input3", { 3.0, 7.0, 1.0, 2.0 } },
                 { "input4", { 8.0, 0.0, -2.0, -3.0 } } },
               { { "output", { 14.0, 15.0, 6.0, 7.0 } } });
}

struct TwoTwoDimInputsFixture : AddNFixture
{
    TwoTwoDimInputsFixture() : AddNFixture({ { 2, 2 }, { 2, 2 } }, 2) {}
};

BOOST_FIXTURE_TEST_CASE(TwoTwoDimInputs, TwoTwoDimInputsFixture)
{
    RunTest<2>({ { "input0", { 1.0, 2.0, 3.0, 4.0 } },
                 { "input1", { 1.0, 5.0, 2.0, 2.0 } } },
               { { "output", { 2.0, 7.0, 5.0, 6.0 } } });
}

struct ThreeTwoDimInputsFixture : AddNFixture
{
    ThreeTwoDimInputsFixture() : AddNFixture({ { 2, 2 }, { 2, 2 }, { 2, 2 } }, 3) {}
};

BOOST_FIXTURE_TEST_CASE(ThreeTwoDimInputs, ThreeTwoDimInputsFixture)
{
    RunTest<2>({ { "input0", { 1.0, 2.0, 3.0, 4.0 } },
                 { "input1", { 1.0, 5.0, 2.0, 2.0 } },
                 { "input2", { 1.0, 1.0, 2.0, 2.0 } } },
               { { "output", { 3.0, 8.0, 7.0, 8.0 } } });
}

struct EightTwoDimInputsFixture : AddNFixture
{
    EightTwoDimInputsFixture() : AddNFixture({ { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 },
                                               { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 } }, 8) {}
};

BOOST_FIXTURE_TEST_CASE(EightTwoDimInputs, EightTwoDimInputsFixture)
{
    RunTest<2>({ { "input0", { 1.0, 2.0, 3.0, 4.0 } },
                 { "input1", { 1.0, 5.0, 2.0, 2.0 } },
                 { "input2", { 1.0, 1.0, 2.0, 2.0 } },
                 { "input3", { 3.0, 7.0, 1.0, 2.0 } },
                 { "input4", { 8.0, 0.0, -2.0, -3.0 } },
                 { "input5", {-3.0, 2.0, -1.0, -5.0 } },
                 { "input6", { 1.0, 6.0, 2.0, 2.0 } },
                 { "input7", {-19.0, 7.0, 1.0, -10.0 } } },
               { { "output", {-7.0, 30.0, 8.0, -6.0 } } });
}

struct ThreeInputBroadcast1D4D4DInputsFixture : AddNFixture
{
    ThreeInputBroadcast1D4D4DInputsFixture() : AddNFixture({ { 1 }, { 1, 1, 2, 2 }, { 1, 1, 2, 2 } }, 3) {}
};

BOOST_FIXTURE_TEST_CASE(ThreeInputBroadcast1D4D4DInputs, ThreeInputBroadcast1D4D4DInputsFixture)
{
    RunTest<4>({ { "input0", { 1.0 } },
                 { "input1", { 1.0, 5.0, 2.0, 2.0 } },
                 { "input2", { 1.0, 1.0, 2.0, 2.0 } } },
               { { "output", { 3.0, 7.0, 5.0, 5.0 } } });
}

struct ThreeInputBroadcast4D1D4DInputsFixture : AddNFixture
{
    ThreeInputBroadcast4D1D4DInputsFixture() : AddNFixture({ { 1, 1, 2, 2 }, { 1 }, { 1, 1, 2, 2 } }, 3) {}
};

BOOST_FIXTURE_TEST_CASE(ThreeInputBroadcast4D1D4DInputs, ThreeInputBroadcast4D1D4DInputsFixture)
{
    RunTest<4>({ { "input0", { 1.0, 3.0, 9.0, 4.0 } },
                 { "input1", {-2.0 } },
                 { "input2", { 1.0, 1.0, 2.0, 2.0 } } },
               { { "output", { 0.0, 2.0, 9.0, 4.0 } } });
}

struct ThreeInputBroadcast4D4D1DInputsFixture : AddNFixture
{
    ThreeInputBroadcast4D4D1DInputsFixture() : AddNFixture({ { 1, 1, 2, 2 }, { 1, 1, 2, 2 }, { 1 } }, 3) {}
};

BOOST_FIXTURE_TEST_CASE(ThreeInputBroadcast4D4D1DInputs, ThreeInputBroadcast4D4D1DInputsFixture)
{
    RunTest<4>({ { "input0", { 1.0, 5.0, 2.0, 2.0 } },
                 { "input1", { 1.0, 1.0, 2.0, 2.0 } },
                 { "input2", { 1.0 } } },
               { { "output", { 3.0, 7.0, 5.0, 5.0 } } });
}

BOOST_AUTO_TEST_SUITE_END()
