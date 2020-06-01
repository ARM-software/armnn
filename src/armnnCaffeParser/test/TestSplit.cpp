//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>
#include "armnnCaffeParser/ICaffeParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(CaffeParser)

struct SplitFixture : public armnnUtils::ParserPrototxtFixture<armnnCaffeParser::ICaffeParser>
{
    SplitFixture()
    {
        m_Prototext = R"(
name: "Split"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape: { dim: 1 dim: 1 dim: 1 dim: 1 } }
}
layer {
    name: "split"
    type: "Split"
    bottom: "data"
    top: "split_top0"
    top: "split_top1"
}
layer {
    bottom: "split_top0"
    bottom: "split_top1"
    top: "add"
    name: "add"
    type: "Eltwise"
}
        )";
        SetupSingleInputSingleOutput("data", "add");
    }
};

BOOST_FIXTURE_TEST_CASE(Split, SplitFixture)
{
    RunTest<1>({ 1.0f }, { 2.0f });
}

BOOST_AUTO_TEST_SUITE_END()
