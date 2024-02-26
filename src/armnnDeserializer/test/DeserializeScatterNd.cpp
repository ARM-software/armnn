//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_ScatterNd")
{
struct ScatterNdFixture : public ParserFlatbuffersSerializeFixture
{
    explicit ScatterNdFixture(const std::string& inputShape,
                              const std::string& indicesShape,
                              const std::string& updatesShape,
                              const std::string& outputShape,
                              const std::string& indicesData,
                              const std::string& updatesData,
                              const std::string dataType,
                              const std::string constDataType)
    {
        m_JsonString = R"(
        {
                inputIds: [0],
                outputIds: [4],
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
                                       dataType: "Signed32",
                                   },
                              data_type: )" + constDataType + R"(,
                              data: {
                                  data: )" + indicesData + R"(,
                                    } }
                                },},
                    {
                    layer_type: "ConstantLayer",
                        layer: {
                               base: {
                                  index:2,
                                  layerName: "ConstantLayer",
                                  layerType: "Constant",
                                   outputSlots: [ {
                                    index: 0,
                                    tensorInfo: {
                                        dimensions: )" + updatesShape + R"(,
                                        dataType: )" + dataType + R"(
                                    },
                                  }],
                              },
                              input: {
                              info: {
                                       dimensions: )" + updatesShape + R"(,
                                       dataType: )" + dataType + R"(
                                   },
                              data_type: )" + constDataType + R"(,
                              data: {
                                  data: )" + updatesData + R"(,
                                    } }
                                },},
                    {
                    layer_type: "ScatterNdLayer",
                        layer: {
                              base: {
                                   index: 3,
                                   layerName: "ScatterNdLayer",
                                   layerType: "ScatterNd",
                                   inputSlots: [
                                   {
                                       index: 0,
                                       connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                   },
                                   {
                                       index: 1,
                                       connection: {sourceLayerIndex:1, outputSlotIndex:0 },
                                   },
                                   {
                                       index: 2,
                                       connection: {sourceLayerIndex:2, outputSlotIndex:0 },
                                   }],
                                   outputSlots: [ {
                                          index: 0,
                                          tensorInfo: {
                                               dimensions: )" + outputShape + R"(,
                                               dataType: )" + dataType + R"(

                                   }}]},
                                    descriptor: {
                                        m_Function: Update,
                                        m_InputEnabled: true,
                                        m_Axis: 0,
                                        m_AxisEnabled: false
                                        },
                        }},
                    {
                    layer_type: "OutputLayer",
                    layer: {
                        base:{
                              layerBindingId: 0,
                              base: {
                                    index: 4,
                                    layerName: "OutputLayer",
                                    layerType: "Output",
                                    inputSlots: [{
                                        index: 0,
                                        connection: {sourceLayerIndex:3, outputSlotIndex:0 },
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

struct SimpleScatterNdFixtureSigned32 : ScatterNdFixture
{
    SimpleScatterNdFixtureSigned32() : ScatterNdFixture("[ 5 ]", "[ 3, 1 ]", "[ 3 ]", "[ 5 ]",
                                                       "[ 0, 1, 2 ]", "[ 1, 2, 3 ]", "Signed32", "IntData") {}
};

TEST_CASE_FIXTURE(SimpleScatterNdFixtureSigned32, "ScatterNdSigned32")
{
    RunTest<1, armnn::DataType::Signed32>(0,
                                         {{"InputLayer", {  0, 0, 0, 0, 0 }}},
                                         {{"OutputLayer", { 1, 2, 3, 0, 0 }}});
}

}

