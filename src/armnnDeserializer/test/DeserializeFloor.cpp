//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_Floor")
{
struct FloorFixture : public ParserFlatbuffersSerializeFixture
{
    explicit FloorFixture(const std::string& shape,
                          const std::string& dataType)
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
                                        dimensions: )" + shape + R"(,
                                        dataType: )" + dataType + R"(
                                        }}]
                                }
                }}},
                {
                layer_type: "FloorLayer",
                layer: {
                      base: {
                           index: 1,
                           layerName: "FloorLayer",
                           layerType: "Floor",
                           inputSlots: [{
                                  index: 0,
                                  connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                           }],
                           outputSlots: [ {
                                  index: 0,
                                  tensorInfo: {
                                       dimensions: )" + shape + R"(,
                                       dataType: )" + dataType + R"(

                           }}]},

                }},
                {
                layer_type: "OutputLayer",
                layer: {
                    base:{
                          layerBindingId: 2,
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
                                        dimensions: )" + shape + R"(,
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


struct SimpleFloorFixture : FloorFixture
{
    SimpleFloorFixture() : FloorFixture("[ 1, 3, 3, 1 ]",
                                        "Float32") {}
};

TEST_CASE_FIXTURE(SimpleFloorFixture, "Floor")
{
    RunTest<4, armnn::DataType::Float32>(
            4,
            {{"InputLayer", { -37.5f, -15.2f, -8.76f, -2.0f, -1.5f, -1.3f, -0.5f, -0.4f, 0.0f}}},
            {{"OutputLayer",{ -38.0f, -16.0f, -9.0f,  -2.0f, -2.0f, -2.0f, -1.0f, -1.0f, 0.0f}}});
}


}
