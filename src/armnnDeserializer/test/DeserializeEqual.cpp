//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersSerializeFixture.hpp"
#include "../Deserializer.hpp"

#include <string>
#include <iostream>

BOOST_AUTO_TEST_SUITE(Deserializer)

struct EqualFixture : public ParserFlatbuffersSerializeFixture
{
    explicit EqualFixture(const std::string & inputShape1,
                          const std::string & inputShape2,
                          const std::string & outputShape,
                          const std::string & dataType)
    {
        m_JsonString = R"(
            {
                inputIds: [0, 1],
                outputIds: [3],
                layers: [
                    {
                        layer_type: "InputLayer",
                        layer: {
                            base: {
                                layerBindingId: 0,
                                base: {
                                    index: 0,
                                    layerName: "InputLayer1",
                                    layerType: "Input",
                                    inputSlots: [{
                                        index: 0,
                                        connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                    }],
                                    outputSlots: [{
                                        index: 0,
                                        tensorInfo: {
                                            dimensions: )" + inputShape1 + R"(,
                                            dataType: )" + dataType + R"(
                                        },
                                    }],
                                },
                            }
                        },
                    },
                    {
                        layer_type: "InputLayer",
                        layer: {
                            base: {
                                layerBindingId: 1,
                                base: {
                                      index:1,
                                      layerName: "InputLayer2",
                                      layerType: "Input",
                                      inputSlots: [{
                                          index: 0,
                                          connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                      }],
                                      outputSlots: [{
                                          index: 0,
                                          tensorInfo: {
                                              dimensions: )" + inputShape2 + R"(,
                                              dataType: )" + dataType + R"(
                                          },
                                      }],
                                },
                            }
                        },
                    },
                    {
                        layer_type: "EqualLayer",
                        layer: {
                            base: {
                                 index:2,
                                 layerName: "EqualLayer",
                                 layerType: "Equal",
                                 inputSlots: [{
                                     index: 0,
                                     connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                 },
                                 {
                                     index: 1,
                                     connection: {sourceLayerIndex:1, outputSlotIndex:0 },
                                 }],
                                 outputSlots: [{
                                     index: 0,
                                     tensorInfo: {
                                         dimensions: )" + outputShape + R"(,
                                         dataType: Boolean
                                     },
                                 }],
                            }
                        },
                    },
                    {
                        layer_type: "OutputLayer",
                        layer: {
                            base:{
                                layerBindingId: 0,
                                base: {
                                    index: 3,
                                    layerName: "OutputLayer",
                                    layerType: "Output",
                                    inputSlots: [{
                                        index: 0,
                                        connection: {sourceLayerIndex:2, outputSlotIndex:0 },
                                    }],
                                    outputSlots: [{
                                        index: 0,
                                        tensorInfo: {
                                            dimensions: )" + outputShape + R"(,
                                            dataType: Boolean
                                        },
                                    }],
                                }
                            }
                        },
                    }
                ]
            }
        )";
        Setup();
    }
};

struct SimpleEqualFixture : EqualFixture
{
    SimpleEqualFixture() : EqualFixture("[ 2, 2, 2, 1 ]",
                                        "[ 2, 2, 2, 1 ]",
                                        "[ 2, 2, 2, 1 ]",
                                        "QuantisedAsymm8") {}
};

BOOST_FIXTURE_TEST_CASE(EqualQuantisedAsymm8, SimpleEqualFixture)
{
  RunTest<4, armnn::DataType::QuantisedAsymm8, armnn::DataType::Boolean>(
      0,
      {{"InputLayer1", { 0, 1, 2, 3, 4, 5, 6, 7 }},
       {"InputLayer2", { 0, 0, 0, 3, 0, 0, 6, 7 }}},
      {{"OutputLayer", { 1, 0, 0, 1, 0, 0, 1, 1 }}});
}

BOOST_AUTO_TEST_SUITE_END()
