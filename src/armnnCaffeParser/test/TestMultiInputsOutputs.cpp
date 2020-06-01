//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>
#include "armnnCaffeParser/ICaffeParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(CaffeParser)

struct MultiInputsOutputsFixture : public armnnUtils::ParserPrototxtFixture<armnnCaffeParser::ICaffeParser>
{
    MultiInputsOutputsFixture()
    {
        m_Prototext = R"(
name: "MultiInputsOutputs"
layer {
  name: "input1"
  type: "Input"
  top: "input1"
  input_param { shape: { dim: 1 } }
}
layer {
  name: "input2"
  type: "Input"
  top: "input2"
  input_param { shape: { dim: 1 } }
}
layer {
    bottom: "input1"
    bottom: "input2"
    top: "add1"
    name: "add1"
    type: "Eltwise"
}
layer {
    bottom: "input2"
    bottom: "input1"
    top: "add2"
    name: "add2"
    type: "Eltwise"
}
        )";
        Setup({ }, { "add1", "add2" });
    }
};

BOOST_FIXTURE_TEST_CASE(MultiInputsOutputs, MultiInputsOutputsFixture)
{
    RunTest<1>({ { "input1",{ 12.0f } },{ "input2",{ 13.0f } } },
        { { "add1",{ 25.0f } },{ "add2",{ 25.0f } } });
}

BOOST_AUTO_TEST_SUITE_END()
