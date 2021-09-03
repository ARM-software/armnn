//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_ChannelShuffle")
{
struct ChannelShuffleFixture : public ParserFlatbuffersSerializeFixture
{
    explicit ChannelShuffleFixture()
    {
        m_JsonString = R"(
        {
          layers: [
            {
              layer_type: "InputLayer",
              layer: {
                base: {
                  base: {
                    layerName: "InputLayer",
                    layerType: "Input",
                    inputSlots: [

                    ],
                    outputSlots: [
                      {
                        tensorInfo: {
                          dimensions: [
                            3,
                            12
                          ],
                          dataType: "Float32",
                          quantizationScale: 0.0,
                          dimensionSpecificity: [
                            true,
                            true
                          ]
                        }
                      }
                    ]
                  }
                }
              }
            },
            {
              layer_type: "ChannelShuffleLayer",
              layer: {
                base: {
                  index: 1,
                  layerName: "channelShuffle",
                  layerType: "ChannelShuffle",
                  inputSlots: [
                    {
                      connection: {
                        sourceLayerIndex: 0,
                        outputSlotIndex: 0
                      }
                    }
                  ],
                  outputSlots: [
                    {
                      tensorInfo: {
                        dimensions: [
                          3,
                          12
                        ],
                        dataType: "Float32",
                        quantizationScale: 0.0,
                        dimensionSpecificity: [
                          true,
                          true
                        ]
                      }
                    }
                  ]
                },
                descriptor: {
                  axis: 1,
                  numGroups: 3
                }
              }
            },
            {
              layer_type: "OutputLayer",
              layer: {
                base: {
                  base: {
                    index: 2,
                    layerName: "OutputLayer",
                    layerType: "Output",
                    inputSlots: [
                      {
                        connection: {
                          sourceLayerIndex: 1,
                          outputSlotIndex: 0
                        }
                      }
                    ],
                    outputSlots: [

                    ]
                  }
                }
              }
            }
          ],
          inputIds: [
            0
          ],
          outputIds: [
            0
          ],
          featureVersions: {
            bindingIdsScheme: 1,
            weightsLayoutScheme: 1,
            constantTensorsAsInputs: 1
          }
        }
    )";
    SetupSingleInputSingleOutput("InputLayer", "OutputLayer");
    }
};

struct SimpleChannelShuffleFixtureFloat32 : ChannelShuffleFixture
{
    SimpleChannelShuffleFixtureFloat32() : ChannelShuffleFixture(){}
};

TEST_CASE_FIXTURE(SimpleChannelShuffleFixtureFloat32, "ChannelShuffleFloat32")
{
    RunTest<2, armnn::DataType::Float32>(0,
                                         {{"InputLayer",
                                           {  0, 1, 2, 3,        4, 5, 6, 7,       8, 9, 10, 11,
                                            12, 13, 14, 15,   16, 17, 18, 19,   20, 21, 22, 23,
                                            24, 25, 26, 27,   28, 29, 30, 31,   32, 33, 34, 35}}},
                                         {{"OutputLayer",
                                           { 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11,
                                            12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23,
                                            24, 28, 32, 25, 29, 33, 26, 30, 34, 27, 31, 35 }}});
}
}