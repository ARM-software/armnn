//
// Copyright © 2020, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_Fill")
{
struct FillFixture : public ParserFlatbuffersSerializeFixture
{
    explicit FillFixture()
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
                            4
                          ],
                          dataType: "Signed32",
                          quantizationScale: 1.0
                        }
                      }
                    ]
                  }
                }
              }
            },
            {
              layer_type: "FillLayer",
              layer: {
                base: {
                  index: 1,
                  layerName: "FillLayer",
                  layerType: "Fill",
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
                          1,
                          3,
                          3,
                          1
                        ],
                        dataType: "Float32",
                        quantizationScale: 0.0
                      }
                    }
                  ]
                },
                descriptor: {
                  value: 1.0
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
            bindingIdsScheme: 1
          }
        }
    )";
        Setup();
    }
};


struct SimpleFillFixture : FillFixture
{
    SimpleFillFixture() : FillFixture() {}
};

TEST_CASE_FIXTURE(SimpleFillFixture, "Fill")
{
    RunTest<4, armnn::DataType::Signed32, armnn::DataType::Float32>(
            0,
            {{"InputLayer", { 1, 3, 3, 1 }}},
            {{"OutputLayer",{ 1, 1, 1, 1, 1, 1, 1, 1, 1}}});
}

}
