//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_Reshape")
{
struct ReshapeFixture : public ParserFlatbuffersSerializeFixture
{
    explicit ReshapeFixture(const std::string &inputShape,
                            const std::string &targetShape,
                            const std::string &outputShape,
                            const std::string &dataType)
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
                                            }}]
                                    }
                    }}},
                    {
                    layer_type: "ReshapeLayer",
                    layer: {
                          base: {
                               index: 1,
                               layerName: "ReshapeLayer",
                               layerType: "Reshape",
                               inputSlots: [{
                                      index: 0,
                                      connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                               }],
                               outputSlots: [ {
                                      index: 0,
                                      tensorInfo: {
                                           dimensions: )" + inputShape + R"(,
                                           dataType: )" + dataType + R"(

                               }}]},
                          descriptor: {
                               targetShape: )" + targetShape + R"(,
                               }

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
                                        connection: {sourceLayerIndex:0, outputSlotIndex:0 },
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
     SetupSingleInputSingleOutput("InputLayer", "OutputLayer");
    }
};

struct SimpleReshapeFixture : ReshapeFixture
{
    SimpleReshapeFixture() : ReshapeFixture("[ 1, 9 ]", "[ 3, 3 ]", "[ 3, 3 ]",
                                            "QuantisedAsymm8") {}
};

struct SimpleReshapeFixture2 : ReshapeFixture
{
    SimpleReshapeFixture2() : ReshapeFixture("[ 2, 2, 1, 1 ]",
                                             "[ 2, 2, 1, 1 ]",
                                             "[ 2, 2, 1, 1 ]",
                                             "Float32") {}
};

TEST_CASE_FIXTURE(SimpleReshapeFixture, "ReshapeQuantisedAsymm8")
{
    RunTest<2, armnn::DataType::QAsymmU8>(0,
                                                { 1, 2, 3, 4, 5, 6, 7, 8, 9 },
                                                { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
}

TEST_CASE_FIXTURE(SimpleReshapeFixture2, "ReshapeFloat32")
{
    RunTest<4, armnn::DataType::Float32>(0,
                                        { 111, 85, 226, 3 },
                                        { 111, 85, 226, 3 });
}


}