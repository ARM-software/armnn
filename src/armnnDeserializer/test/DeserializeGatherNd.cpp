//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_GatherNd")
{
struct GatherNdFixture : public ParserFlatbuffersSerializeFixture
{
    explicit GatherNdFixture(const std::string& paramsShape,
                             const std::string& indicesShape,
                             const std::string& outputShape,
                             const std::string& indicesData,
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
                                            dimensions: )" + paramsShape + R"(,
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
                                  data: )" + indicesData + R"(,
                                    } }
                                },},
                    {
                    layer_type: "GatherNdLayer",
                        layer: {
                              base: {
                                   index: 2,
                                   layerName: "GatherNdLayer",
                                   layerType: "GatherNd",
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
                }],
                featureVersions: {
                    weightsLayoutScheme: 1,
                }
                 } )";

        Setup();
    }
};

struct SimpleGatherNdFixtureFloat32 : GatherNdFixture
{
    SimpleGatherNdFixtureFloat32() : GatherNdFixture("[ 6, 3 ]", "[ 3, 1 ]", "[ 3, 3 ]",
                                                     "[ 5, 1, 0 ]", "Float32", "IntData") {}
};

TEST_CASE_FIXTURE(SimpleGatherNdFixtureFloat32, "GatherNdFloat32")
{
    RunTest<4, armnn::DataType::Float32>(0,
                                         {{"InputLayer", {  1,  2,  3,
                                                            4,  5,  6,
                                                            7,  8,  9,
                                                            10, 11, 12,
                                                            13, 14, 15,
                                                            16, 17, 18 }}},
                                         {{"OutputLayer", { 16, 17, 18,
                                                            4,  5,  6,
                                                            1,  2,  3}}});
}

}

