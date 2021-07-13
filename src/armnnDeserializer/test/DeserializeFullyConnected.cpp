//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("DeserializeParser_FullyConnected")
{
struct FullyConnectedFixture : public ParserFlatbuffersSerializeFixture
{
    explicit FullyConnectedFixture(const std::string & inputShape1,
                                   const std::string & outputShape,
                                   const std::string & weightsShape,
                                   const std::string & dataType)
    {
        m_JsonString = R"(
        {
            inputIds: [0],
            outputIds: [2],
            layers: [{
                layer_type: "InputLayer",
                layer: {
                    base: {
                        layerBindingId: 0,
                        base: {
                            index: 0,
                            layerName: "InputLayer",
                            layerType: "Input",
                            inputSlots: [{
                                index: 0,
                                connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                }],
                            outputSlots: [{
                                index: 0,
                                tensorInfo: {
                                    dimensions: )" + inputShape1 + R"(,
                                    dataType: )" + dataType + R"(,
                                    quantizationScale: 1.0,
                                    quantizationOffset: 0
                                    },
                                }]
                            },
                        }
                    },
                },
            {
            layer_type: "FullyConnectedLayer",
            layer : {
                base: {
                    index:1,
                    layerName: "FullyConnectedLayer",
                    layerType: "FullyConnected",
                    inputSlots: [{
                            index: 0,
                            connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                        }],
                    outputSlots: [{
                        index: 0,
                        tensorInfo: {
                            dimensions: )" + outputShape + R"(,
                            dataType: )" + dataType + R"(,
                            quantizationScale: 2.0,
                            quantizationOffset: 0
                        },
                        }],
                    },
                descriptor: {
                    biasEnabled: false,
                    transposeWeightsMatrix: true
                    },
                weights: {
                    info: {
                             dimensions: )" + weightsShape + R"(,
                             dataType: )" + dataType + R"(,
                             quantizationScale: 1.0,
                             quantizationOffset: 0
                         },
                    data_type: ByteData,
                    data: {
                        data: [
                            2, 3, 4, 5
                            ],
                        }
                    }
                },
            },
            {
            layer_type: "OutputLayer",
            layer: {
                base:{
                    layerBindingId: 0,
                    base: {
                        index: 2,
                        layerName: "OutputLayer",
                        layerType: "Output",
                        inputSlots: [{
                            index: 0,
                            connection: {sourceLayerIndex:1, outputSlotIndex:0 },
                        }],
                        outputSlots: [ {
                            index: 0,
                            tensorInfo: {
                                dimensions: )" + outputShape + R"(,
                                dataType: )" + dataType + R"(
                            },
                        }],
                    }
                }},
            }]
        }
        )";
        Setup();
    }
};


struct FullyConnectedFixtureConstantAsInput : public ParserFlatbuffersSerializeFixture
{
    explicit FullyConnectedFixtureConstantAsInput()
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
                        4,
                        1,
                        1
                      ],
                      "dataType": "QAsymmU8",
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
          "layer_type": "FullyConnectedLayer",
          "layer": {
            "base": {
              "index": 1,
              "layerName": "FullyConnectedLayer",
              "layerType": "FullyConnected",
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
                      1
                    ],
                    "dataType": "QAsymmU8",
                    "quantizationScale": 2.0,
                    "quantizationOffset": 0,
                    "quantizationDim": 0,
                    "dimensionality": 1,
                    "dimensionSpecificity": [
                      true,
                      true
                    ]
                  }
                }
              ]
            },
            "descriptor": {
              "biasEnabled": false,
              "transposeWeightsMatrix": true,
              "constantWeights": true
            }
          }
        },
        {
          "layer_type": "ConstantLayer",
          "layer": {
            "base": {
              "index": 2,
              "layerName": "",
              "layerType": "Constant",
              "inputSlots": [

              ],
              "outputSlots": [
                {
                  "index": 0,
                  "tensorInfo": {
                    "dimensions": [
                      1,
                      4
                    ],
                    "dataType": "QAsymmU8",
                    "quantizationScale": 1.0,
                    "quantizationOffset": 0,
                    "quantizationDim": 0,
                    "dimensionality": 1,
                    "dimensionSpecificity": [
                      true,
                      true
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
                  4
                ],
                "dataType": "QAsymmU8",
                "quantizationScale": 1.0,
                "quantizationOffset": 0,
                "quantizationDim": 0,
                "dimensionality": 1,
                "dimensionSpecificity": [
                  true,
                  true
                ]
              },
              "data_type": "ByteData",
              "data": {
                "data": [
                  2,
                  3,
                  4,
                  5
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
        "weightsLayoutScheme": 1,
        "constantTensorsAsInputs": 1
      }
    }
    )";
        Setup();
    }
};

struct FullyConnectedWithNoBiasFixture : FullyConnectedFixture
{
    FullyConnectedWithNoBiasFixture()
            : FullyConnectedFixture("[ 1, 4, 1, 1 ]",     // inputShape
                                    "[ 1, 1 ]",           // outputShape
                                    "[ 1, 4 ]",           // filterShape
                                    "QuantisedAsymm8")    // filterData
    {}
};

TEST_CASE_FIXTURE(FullyConnectedWithNoBiasFixture, "FullyConnectedWithNoBias")
{
    // Weights and biases used to be always constant and were stored as members of the layer. This has changed and
    // they are now passed as inputs (ConstantLayer) but the old way can still be used for now.
    RunTest<2, armnn::DataType::QAsymmU8>(
            0,
            {{"InputLayer",  { 10, 20, 30, 40 }}},
            {{"OutputLayer", { 400/2 }}});
}

struct FullyConnectedWithNoBiasFixtureConstantAsInput : FullyConnectedFixtureConstantAsInput
{
    FullyConnectedWithNoBiasFixtureConstantAsInput()
            : FullyConnectedFixtureConstantAsInput()
    {}
};

TEST_CASE_FIXTURE(FullyConnectedWithNoBiasFixtureConstantAsInput, "FullyConnectedWithNoBiasConstantAsInput")
{
    RunTest<2, armnn::DataType::QAsymmU8>(
            0,
            {{"InputLayer",  { 10, 20, 30, 40 }}},
            {{"OutputLayer", { 400/2 }}});
}

}
