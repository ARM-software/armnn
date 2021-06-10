//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_Rsqrt")
{
struct RsqrtFixture : public ParserFlatbuffersSerializeFixture
{
    explicit RsqrtFixture(const std::string & inputShape,
                          const std::string & outputShape,
                          const std::string & dataType)
    {
        m_JsonString = R"(
        {
                inputIds: [0],
                outputIds: [2],
                layers: [
                {
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
                                    outputSlots: [ {
                                        index: 0,
                                        tensorInfo: {
                                            dimensions: )" + inputShape + R"(,
                                            dataType: )" + dataType + R"(
                                        },
                                    }],
                                 },}},
                },
                {
                layer_type: "RsqrtLayer",
                layer : {
                        base: {
                             index:1,
                             layerName: "RsqrtLayer",
                             layerType: "Rsqrt",
                             inputSlots: [
                                            {
                                             index: 0,
                                             connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                            }
                             ],
                             outputSlots: [ {
                                 index: 0,
                                 tensorInfo: {
                                     dimensions: )" + outputShape + R"(,
                                     dataType: )" + dataType + R"(
                                 },
                             }],
                            }},
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
                            }}},
                }]
         }
        )";
        Setup();
    }
};


struct Rsqrt2dFixture : RsqrtFixture
{
    Rsqrt2dFixture() : RsqrtFixture("[ 2, 2 ]",
                                    "[ 2, 2 ]",
                                    "Float32") {}
};

TEST_CASE_FIXTURE(Rsqrt2dFixture, "Rsqrt2d")
{
  RunTest<2, armnn::DataType::Float32>(
      0,
      {{"InputLayer", { 1.0f,  4.0f,
                        16.0f, 25.0f }}},
      {{"OutputLayer",{ 1.0f,  0.5f,
                        0.25f, 0.2f }}});
}


}
