//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_Gather")
{
struct GatherFixture : public ParserFlatbuffersSerializeFixture
{
    explicit GatherFixture(const std::string& inputShape,
                           const std::string& indicesShape,
                           const std::string& input1Content,
                           const std::string& outputShape,
                           const std::string& axis,
                           const std::string dataType,
                           const std::string constDataType)
    {
        m_JsonString = R"(
        {
                inputIds: [0],
                outputIds: [3],
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
                                            }}]
                                    }
                    }}},
                    {
                    layer_type: "ConstantLayer",
                        layer: {
                               base: {
                                  index:1,
                                  layerName: "ConstantLayer",
                                  layerType: "Constant",
                                   outputSlots: [ {
                                    index: 0,
                                    tensorInfo: {
                                        dimensions: )" + indicesShape + R"(,
                                        dataType: "Signed32",
                                    },
                                  }],
                              },
                              input: {
                              info: {
                                       dimensions: )" + indicesShape + R"(,
                                       dataType: )" + dataType + R"(
                                   },
                              data_type: )" + constDataType + R"(,
                              data: {
                                  data: )" + input1Content + R"(,
                                    } }
                                },},
                    {
                    layer_type: "GatherLayer",
                        layer: {
                              base: {
                                   index: 2,
                                   layerName: "GatherLayer",
                                   layerType: "Gather",
                                   inputSlots: [
                                   {
                                       index: 0,
                                       connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                   },
                                   {
                                        index: 1,
                                        connection: {sourceLayerIndex:1, outputSlotIndex:0 }
                                   }],
                                   outputSlots: [ {
                                          index: 0,
                                          tensorInfo: {
                                               dimensions: )" + outputShape + R"(,
                                               dataType: )" + dataType + R"(

                                   }}]},
                                   descriptor: {
                                       axis: )" + axis + R"(
                                   }
                        }},
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
                                    outputSlots: [ {
                                        index: 0,
                                        tensorInfo: {
                                            dimensions: )" + outputShape + R"(,
                                            dataType: )" + dataType + R"(
                                        },
                                }],
                            }}},
                }]
                 } )";

        Setup();
    }
};

struct SimpleGatherFixtureFloat32 : GatherFixture
{
    SimpleGatherFixtureFloat32() : GatherFixture("[ 3, 2, 3 ]", "[ 2, 3 ]", "[1, 2, 1, 2, 1, 0]",
                                                 "[ 2, 3, 2, 3 ]", "0", "Float32", "IntData") {}
};

TEST_CASE_FIXTURE(SimpleGatherFixtureFloat32, "GatherFloat32")
{
    RunTest<4, armnn::DataType::Float32>(0,
                                         {{"InputLayer", {  1,  2,  3,
                                                            4,  5,  6,
                                                            7,  8,  9,
                                                            10, 11, 12,
                                                            13, 14, 15,
                                                            16, 17, 18 }}},
                                         {{"OutputLayer", { 7,  8,  9,
                                                            10, 11, 12,
                                                            13, 14, 15,
                                                            16, 17, 18,
                                                            7,  8,  9,
                                                            10, 11, 12,
                                                            13, 14, 15,
                                                            16, 17, 18,
                                                            7,  8,  9,
                                                            10, 11, 12,
                                                            1,  2,  3,
                                                            4,  5,  6 }}});
}

}

