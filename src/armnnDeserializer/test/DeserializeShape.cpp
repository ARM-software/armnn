//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <doctest/doctest.h>

#include <string>

TEST_SUITE("Deserializer_Shape")
{
struct ShapeFixture : public ParserFlatbuffersSerializeFixture
{
    explicit ShapeFixture()
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
                                1,
                                3,
                                3,
                                1
                              ],
                              dataType: "Signed32",
                              quantizationScale: 0.0
                            }
                          }
                        ]
                      }
                    }
                  }
                },
                {
                  layer_type: "ShapeLayer",
                  layer: {
                    base: {
                      index: 1,
                      layerName: "shape",
                      layerType: "Shape",
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
                              4
                            ],
                            dataType: "Signed32",
                            quantizationScale: 0.0
                          }
                        }
                      ]
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

struct SimpleShapeFixture : ShapeFixture
{
    SimpleShapeFixture() : ShapeFixture() {}
};

TEST_CASE_FIXTURE(SimpleShapeFixture, "DeserializeShape")
{
    RunTest<1, armnn::DataType::Signed32>(
            0,
            {{"InputLayer", { 1, 1, 1, 1, 1, 1, 1, 1, 1 }}},
            {{"OutputLayer",{ 1, 3, 3, 1 }}});
}

}
