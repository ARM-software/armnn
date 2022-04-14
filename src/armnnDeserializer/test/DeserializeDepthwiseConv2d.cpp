//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"

#include <armnnDeserializer/IDeserializer.hpp>

#include <doctest/doctest.h>

#include <string>

TEST_SUITE("Deserializer_DepthwiseConv2d")
{
struct DepthwiseConv2dFlatbufferVersion1FixtureOld : public ParserFlatbuffersSerializeFixture
{
    explicit DepthwiseConv2dFlatbufferVersion1FixtureOld()
    {
        m_JsonString = R"(
        {
          "layers": [
            {
              "layer_type": "InputLayer",
              "layer": {
                "base": {
                  "base": {
                    "index": 0,
                    "layerName": "Input",
                    "layerType": "Input",
                    "inputSlots": [

                    ],
                    "outputSlots": [
                      {
                        "index": 0,
                        "tensorInfo": {
                          "dimensions": [
                            1,
                            3,
                            3,
                            3
                          ],
                          "dataType": "QAsymmS8",
                          "quantizationScale": 1.0,
                          "quantizationOffset": 0,
                          "quantizationDim": 0,
                          "dimensionality": 1,
                          "dimensionSpecificity": [
                            true,
                            true,
                            true,
                            true
                          ]
                        }
                      }
                    ]
                  },
                  "layerBindingId": 0
                }
              }
            },
            {
              "layer_type": "DepthwiseConvolution2dLayer",
              "layer": {
                "base": {
                  "index": 1,
                  "layerName": "depwiseConvolution2dWithPerAxis",
                  "layerType": "DepthwiseConvolution2d",
                  "inputSlots": [
                    {
                      "index": 0,
                      "connection": {
                        "sourceLayerIndex": 0,
                        "outputSlotIndex": 0
                      }
                    }
                  ],
                  "outputSlots": [
                    {
                      "index": 0,
                      "tensorInfo": {
                        "dimensions": [
                          1,
                          3,
                          3,
                          3
                        ],
                        "dataType": "QAsymmS8",
                        "quantizationScale": 1.0,
                        "quantizationOffset": 0,
                        "quantizationDim": 0,
                        "dimensionality": 1,
                        "dimensionSpecificity": [
                          true,
                          true,
                          true,
                          true
                        ]
                      }
                    }
                  ]
                },
                "descriptor": {
                  "padLeft": 1,
                  "padRight": 1,
                  "padTop": 1,
                  "padBottom": 1,
                  "strideX": 1,
                  "strideY": 1,
                  "dilationX": 1,
                  "dilationY": 1,
                  "biasEnabled": false,
                  "dataLayout": "NHWC"
                },
                "weights": {
                  "info": {
                    "dimensions": [
                      1,
                      3,
                      3,
                      3
                    ],
                    "dataType": "QSymmS8",
                    "quantizationScale": 0.25,
                    "quantizationOffset": 0,
                    "quantizationScales": [
                      0.25,
                      0.2,
                      0.1
                    ],
                    "quantizationDim": 0,
                    "dimensionality": 1,
                    "dimensionSpecificity": [
                      true,
                      true,
                      true,
                      true
                    ]
                  },
                  "data_type": "ByteData",
                  "data": {
                    "data": [
                      4,
                      20,
                      0,
                      8,
                      20,
                      30,
                      4,
                      0,
                      10,
                      12,
                      0,
                      40,
                      0,
                      5,
                      30,
                      16,
                      10,
                      40,
                      12,
                      0,
                      30,
                      16,
                      20,
                      0,
                      12,
                      20,
                      20
                    ]
                  }
                }
              }
            },
            {
              "layer_type": "OutputLayer",
              "layer": {
                "base": {
                  "base": {
                    "index": 2,
                    "layerName": "Output",
                    "layerType": "Output",
                    "inputSlots": [
                      {
                        "index": 0,
                        "connection": {
                          "sourceLayerIndex": 1,
                          "outputSlotIndex": 0
                        }
                      }
                    ],
                    "outputSlots": [

                    ]
                  },
                  "layerBindingId": 0
                }
              }
            }
          ],
          "inputIds": [
            0
          ],
          "outputIds": [
            0
          ],
          "featureVersions": {
            "bindingIdsScheme": 1
          }
        }
        )";
        SetupSingleInputSingleOutput("Input", "Output");
    }
};

struct DepthwiseConv2dFlatbufferVersion1Fixture : public ParserFlatbuffersSerializeFixture
{
    explicit DepthwiseConv2dFlatbufferVersion1Fixture()
    {
        m_JsonString = R"(
        {
          "layers": [
            {
              "layer_type": "InputLayer",
              "layer": {
                "base": {
                  "base": {
                    "index": 0,
                    "layerName": "InputLayer",
                    "layerType": "Input",
                    "inputSlots": [
                    ],
                    "outputSlots": [
                      {
                        "index": 0,
                        "tensorInfo": {
                          "dimensions": [
                            1,
                            3,
                            3,
                            3
                          ],
                          "dataType": "QAsymmS8",
                          "quantizationScale": 1.0,
                          "quantizationOffset": 0,
                          "quantizationDim": 0,
                          "dimensionality": 1,
                          "dimensionSpecificity": [
                            true,
                            true,
                            true,
                            true
                          ]
                        }
                      }
                    ]
                  },
                  "layerBindingId": 0
                }
              }
            },
            {
              "layer_type": "DepthwiseConvolution2dLayer",
              "layer": {
                "base": {
                  "index": 1,
                  "layerName": "depthwiseConvolution2dWithPerAxis",
                  "layerType": "DepthwiseConvolution2d",
                  "inputSlots": [
                    {
                      "index": 0,
                      "connection": {
                        "sourceLayerIndex": 0,
                        "outputSlotIndex": 0
                      }
                    },
                    {
                      "index": 1,
                      "connection": {
                        "sourceLayerIndex": 2,
                        "outputSlotIndex": 0
                      }
                    }
                  ],
                  "outputSlots": [
                    {
                      "index": 0,
                      "tensorInfo": {
                        "dimensions": [
                          1,
                          3,
                          3,
                          3
                        ],
                        "dataType": "QAsymmS8",
                        "quantizationScale": 1.0,
                        "quantizationOffset": 0,
                        "quantizationDim": 0,
                        "dimensionality": 1,
                        "dimensionSpecificity": [
                          true,
                          true,
                          true,
                          true
                        ]
                      }
                    }
                  ]
                },
                "descriptor": {
                  "padLeft": 1,
                  "padRight": 1,
                  "padTop": 1,
                  "padBottom": 1,
                  "strideX": 1,
                  "strideY": 1,
                  "dilationX": 1,
                  "dilationY": 1,
                  "biasEnabled": false,
                  "dataLayout": "NHWC"
                }
              }
            },
            {
              "layer_type": "ConstantLayer",
              "layer": {
                "base": {
                  "index": 2,
                  "layerName": "Weights",
                  "layerType": "Constant",
                  "inputSlots": [
                  ],
                  "outputSlots": [
                    {
                      "index": 0,
                      "tensorInfo": {
                        "dimensions": [
                          1,
                          3,
                          3,
                          3
                        ],
                        "dataType": "QSymmS8",
                        "quantizationScale": 0.25,
                        "quantizationOffset": 0,
                        "quantizationDim": 0,
                        "dimensionality": 1,
                        "dimensionSpecificity": [
                          true,
                          true,
                          true,
                          true
                        ],
                        quantizationScales: [
                              0.25,
                              0.2,
                              0.1
                        ],
                        "isConstant": true,
                      }
                    }
                  ]
                },
                "input": {
                  "info": {
                    "dimensions": [
                       1,
                       3,
                       3,
                       3
                    ],
                    "dataType": "QSymmS8",
                    "quantizationScale": 0.25,
                    "quantizationOffset": 0,
                    "quantizationDim": 0,
                    "dimensionality": 1,
                    "dimensionSpecificity": [
                      true,
                      true,
                      true,
                      true
                    ],
                    quantizationScales: [
                      0.25,
                      0.2,
                      0.1
                    ]
                  },
                  "data_type": "ByteData",
                  "data": {
                    "data": [
                      4,
                      20,
                      0,
                      8,
                      20,
                      30,
                      4,
                      0,
                      10,
                      12,
                      0,
                      40,
                      0,
                      5,
                      30,
                      16,
                      10,
                      40,
                      12,
                      0,
                      30,
                      16,
                      20,
                      0,
                      12,
                      20,
                      20
                    ]
                  }
                }
              }
            },
            {
              "layer_type": "OutputLayer",
              "layer": {
                "base": {
                  "base": {
                    "index": 3,
                    "layerName": "OutputLayer",
                    "layerType": "Output",
                    "inputSlots": [
                      {
                        "index": 0,
                        "connection": {
                          "sourceLayerIndex": 1,
                          "outputSlotIndex": 0
                        }
                      }
                    ],
                    "outputSlots": [
                    ]
                  },
                  "layerBindingId": 0
                }
              }
            }
          ],
          "inputIds": [
            0
          ],
          "outputIds": [
            0
          ],
          "featureVersions": {
            "bindingIdsScheme": 1,
            "constantTensorsAsInputs": 1
          }
        }
        )";
        Setup();
    }
};

// This test uses a model that was created before weights layout scheme version was added to our flatbuffers
// file. It ensures older models can still be read and executed
// featureVersion weights layout scheme 1 indicates a change in the depthwise weights layout within
// armm from [M,I,H,W] --> [1,H,W,I*M]
TEST_CASE_FIXTURE(DepthwiseConv2dFlatbufferVersion1FixtureOld, "DepthwiseConv2d_FlatbufferVersion1Old")
{
    RunTest<4, armnn::DataType::QAsymmS8>(
        0,
        { 3,2,0,0,4,3,0,1,2,
          0,1,3,0,4,2,2,2,3,
          2,4,3,2,0,4,3,4,0},
        { 15,60,10,11,37,20, 0,18,17,
          20,65,28,28,74,26,12,20,18,
          25,36,12,37,42,25,29,14, 9});
}

TEST_CASE_FIXTURE(DepthwiseConv2dFlatbufferVersion1Fixture,
                  "DepthwiseConv2d_FlatbufferVersion1_WeightsAndBiasesAsConstantLayers")
{
    RunTest<4, armnn::DataType::QAsymmS8>(
            0,
            {{"InputLayer", { 3,2,0,0,4,3,0,1,2,
                              0,1,3,0,4,2,2,2,3,
                              2,4,3,2,0,4,3,4,0}}},
            {{"OutputLayer", { 15,60,10,11,37,20, 0,18,17,
                               20,65,28,28,74,26,12,20,18,
                               25,36,12,37,42,25,29,14, 9}}});
}

}