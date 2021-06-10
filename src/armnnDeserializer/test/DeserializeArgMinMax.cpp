//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include "../Deserializer.hpp"

#include <string>
#include <iostream>

TEST_SUITE("DeserializeParser_ArgMinMax")
{
struct ArgMinMaxFixture : public ParserFlatbuffersSerializeFixture
{
    explicit ArgMinMaxFixture(const std::string& inputShape,
                              const std::string& outputShape,
                              const std::string& axis,
                              const std::string& argMinMaxFunction)
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
                          dimensions: )" + inputShape + R"(,
                          dataType: "Float32",
                          quantizationScale: 0.0
                        }
                      }
                    ]
                  }
                }
              }
            },
            {
              layer_type: "ArgMinMaxLayer",
              layer: {
                base: {
                  index: 1,
                  layerName: "ArgMinMaxLayer",
                  layerType: "ArgMinMax",
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
                        dimensions: )" + outputShape + R"(,
                        dataType: "Signed64",
                        quantizationScale: 0.0
                      }
                    }
                  ]
                },
                descriptor: {
                  axis: )" + axis + R"(,
                  argMinMaxFunction: )" + argMinMaxFunction + R"(
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

struct SimpleArgMinMaxFixture : public ArgMinMaxFixture
{
    SimpleArgMinMaxFixture() : ArgMinMaxFixture("[ 1, 1, 1, 5 ]",
                                                "[ 1, 1, 1 ]",
                                                "-1",
                                                "Max") {}
};

TEST_CASE_FIXTURE(SimpleArgMinMaxFixture, "ArgMinMax")
{
    RunTest<3, armnn::DataType::Float32, armnn::DataType::Signed64>(
            0,
            {{"InputLayer", { 6.0f, 2.0f, 8.0f, 10.0f, 9.0f}}},
            {{"OutputLayer",{ 3l }}});
}

}
