//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"
#include "test/GraphUtils.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct AssertSimpleFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    AssertSimpleFixture()
    {
        //     Placeholder   AssertInput
        //      |       \     /
        //     Add ------ Assert

        m_Prototext = R"(
            node {
              name: "Placeholder"
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
              name: "AssertInput"
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
              name: "Assert"
              op: "Assert"
              input: "Placeholder"
              input: "AssertInput"
              attr {
                key: "T"
                value {
                  type: DT_FLOAT
                }
              }
            }
            node {
              name: "Add"
              op: "Add"
              input: "Placeholder"
              input: "Placeholder"
              input: "^Assert"
              attr {
                key: "T"
                value {
                  type: DT_FLOAT
                }
              }
            })";
    }
};

BOOST_FIXTURE_TEST_CASE(AssertSimpleTest, AssertSimpleFixture)
{
    SetupSingleInputSingleOutput({ 1, 1, 1, 4 }, "Placeholder", "Add");
    RunTest<4>({ 1.0f, 2.0f, 3.0f, 4.0f }, { 2.0f, 4.0f, 6.0f, 8.0f });
}

BOOST_FIXTURE_TEST_CASE(AssertSimpleGraphStructureTest, AssertSimpleFixture)
{
    auto optimized = SetupOptimizedNetwork({ { "Placeholder", { 1, 1, 1, 4 } } }, { "Add" });

    auto optimizedNetwork = armnn::PolymorphicDowncast<armnn::OptimizedNetwork*>(optimized.get());
    auto graph = optimizedNetwork->GetGraph();

    BOOST_TEST((graph.GetNumInputs() == 1));
    BOOST_TEST((graph.GetNumOutputs() == 1));
    BOOST_TEST((graph.GetNumLayers() == 3));

    armnn::Layer* inputLayer = GetFirstLayerWithName(graph, "Placeholder");
    BOOST_TEST((inputLayer->GetType() == armnn::LayerType::Input));
    BOOST_TEST(CheckNumberOfInputSlot(inputLayer, 0));
    BOOST_TEST(CheckNumberOfOutputSlot(inputLayer, 1));

    armnn::Layer* addLayer = GetFirstLayerWithName(graph, "Add");
    BOOST_TEST((addLayer->GetType() == armnn::LayerType::Addition));
    BOOST_TEST(CheckNumberOfInputSlot(addLayer, 2));
    BOOST_TEST(CheckNumberOfOutputSlot(addLayer, 1));

    armnn::TensorInfo tensorInfo(armnn::TensorShape({1, 1, 1, 4}), armnn::DataType::Float32);
    BOOST_TEST(IsConnected(inputLayer, addLayer, 0, 0, tensorInfo));
    BOOST_TEST(IsConnected(inputLayer, addLayer, 0, 1, tensorInfo));

    for (auto&& outputLayer : graph.GetOutputLayers())
    {
        BOOST_TEST(IsConnected(addLayer, const_cast<armnn::OutputLayer*>(outputLayer), 0, 0, tensorInfo));
    }
}

struct AssertFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    AssertFixture()
    {
        // Input0    Input1  Input2
        //  |    \    /        |
        //  |     Sub ------ Assert
        //   \     /         /
        //    Output -------

        m_Prototext = R"(
            node {
              name: "Input0"
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
              name: "Input1"
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
              name: "Sub"
              op: "Sub"
              input: "Input0"
              input: "Input1"
              attr {
                key: "T"
                value {
                  type: DT_FLOAT
                }
              }
            }
            node {
              name: "Input2"
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
              name: "Assert"
              op: "Assert"
              input: "Input2"
              input: "Sub"
              attr {
                key: "T"
                value {
                  type: DT_FLOAT
                }
              }
            }
            node {
              name: "Output"
              op: "Add"
              input: "Input0"
              input: "Sub"
              input: "^Assert"
              attr {
                key: "T"
                value {
                  type: DT_FLOAT
                }
              }
            })";


    }
};

BOOST_FIXTURE_TEST_CASE(AssertTest, AssertFixture)
{
    Setup({ { "Input0", { 1, 1, 2, 2 } },
            { "Input1", { 1, 1, 2, 2 } } },
          { "Output" });

    RunTest<4>({ { "Input0", { 4.0f,   3.0f,
                               2.0f,   1.0f } },

                 { "Input1", { 1.0f,   2.0f,
                               3.0f,   4.0f } } },

               { { "Output", { 7.0f,   4.0f,
                               1.0f,  -2.0f } } });
}

BOOST_FIXTURE_TEST_CASE(AssertGraphStructureTest, AssertFixture)
{
    auto optimized = SetupOptimizedNetwork({ { "Input0", { 1, 1, 2, 2 } },
                                             { "Input1", { 1, 1, 2, 2 } } },
                                           { "Output" });

    auto optimizedNetwork = armnn::PolymorphicDowncast<armnn::OptimizedNetwork*>(optimized.get());
    auto graph = optimizedNetwork->GetGraph();

    BOOST_TEST((graph.GetNumInputs() == 2));
    BOOST_TEST((graph.GetNumOutputs() == 1));
    BOOST_TEST((graph.GetNumLayers() == 5));

    armnn::Layer* inputLayer0 = GetFirstLayerWithName(graph, "Input0");
    BOOST_TEST((inputLayer0->GetType() == armnn::LayerType::Input));
    BOOST_TEST(CheckNumberOfInputSlot(inputLayer0, 0));
    BOOST_TEST(CheckNumberOfOutputSlot(inputLayer0, 1));

    armnn::Layer* inputLayer1 = GetFirstLayerWithName(graph, "Input1");
    BOOST_TEST((inputLayer1->GetType() == armnn::LayerType::Input));
    BOOST_TEST(CheckNumberOfInputSlot(inputLayer1, 0));
    BOOST_TEST(CheckNumberOfOutputSlot(inputLayer1, 1));

    armnn::Layer* subLayer = GetFirstLayerWithName(graph, "Sub");
    BOOST_TEST((subLayer->GetType() == armnn::LayerType::Subtraction));
    BOOST_TEST(CheckNumberOfInputSlot(subLayer, 2));
    BOOST_TEST(CheckNumberOfOutputSlot(subLayer, 1));

    armnn::Layer* addLayer = GetFirstLayerWithName(graph, "Output");
    BOOST_TEST((addLayer->GetType() == armnn::LayerType::Addition));
    BOOST_TEST(CheckNumberOfInputSlot(addLayer, 2));
    BOOST_TEST(CheckNumberOfOutputSlot(addLayer, 1));

    armnn::TensorInfo tensorInfo(armnn::TensorShape({1, 1, 2, 2}), armnn::DataType::Float32);
    BOOST_TEST(IsConnected(inputLayer0, subLayer, 0, 0, tensorInfo));
    BOOST_TEST(IsConnected(inputLayer1, subLayer, 0, 1, tensorInfo));
    BOOST_TEST(IsConnected(inputLayer0, addLayer, 0, 0, tensorInfo));
    BOOST_TEST(IsConnected(subLayer, addLayer, 0, 1, tensorInfo));

    for (auto&& outputLayer : graph.GetOutputLayers())
    {
        BOOST_TEST(IsConnected(addLayer, const_cast<armnn::OutputLayer*>(outputLayer), 0, 0, tensorInfo));
    }
}


BOOST_AUTO_TEST_SUITE_END()
