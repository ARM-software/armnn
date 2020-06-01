//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>
#include "armnnCaffeParser/ICaffeParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(CaffeParser)

// The pooling layer should take its input from the relu, not the add directly.
struct InPlaceFixture : public armnnUtils::ParserPrototxtFixture<armnnCaffeParser::ICaffeParser>
{
    InPlaceFixture()
    {
        m_Prototext = R"(
name: "InPlace"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape: { dim: 1 dim: 1 dim: 1 dim: 1 } }
}
layer {
    bottom: "data"
    bottom: "data"
    top: "add"
    name: "add"
    type: "Eltwise"
}
layer {
  name: "relu"
  type: "ReLU"
  bottom: "add"
  top: "relu"
  phase: TEST
}
layer {
  name: "pool"
  type: "Pooling"
  bottom: "relu"
  top: "pool"
  phase: TEST
  pooling_param {
    pool: MAX
    kernel_size: 1
    stride: 1
  }
}
        )";
        SetupSingleInputSingleOutput("data", "pool");
    }
};

BOOST_FIXTURE_TEST_CASE(ParseInPlace, InPlaceFixture)
{
    RunTest<1>({ -1.0f }, { 0.0f });
}

// The requested output of the network is a layer which has an activation attached.
// The output of the network should therefore actually be the activation layer.
struct InPlaceOutputFixture : public armnnUtils::ParserPrototxtFixture<armnnCaffeParser::ICaffeParser>
{
    InPlaceOutputFixture()
    {
        m_Prototext = R"(
name: "InPlace"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape: { dim: 1 dim: 1 dim: 1 dim: 1 } }
}
layer {
    bottom: "data"
    bottom: "data"
    top: "add"
    name: "add"
    type: "Eltwise"
}
layer {
  name: "relu"
  type: "ReLU"
  bottom: "add"
  top: "add"
  phase: TEST
}
        )";
        SetupSingleInputSingleOutput("data", "add");
    }
};

BOOST_FIXTURE_TEST_CASE(InPlaceOutput, InPlaceOutputFixture)
{
    RunTest<1>({ -1.0f }, { 0.0f });
}

BOOST_AUTO_TEST_SUITE_END()
