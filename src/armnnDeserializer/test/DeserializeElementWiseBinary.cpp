//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("ElementWiseBinary_Deserializer")
{
    struct ElementwiseBinaryFixture : public ParserFlatbuffersSerializeFixture {
        explicit ElementwiseBinaryFixture(const std::string & inputShape1,
                                          const std::string & inputShape2,
                                          const std::string & outputShape,
                                          const std::string & dataType,
                                          const std::string &binaryOperation) {
            m_JsonString = R"(
            {
                    inputIds: [0, 1],
                    outputIds: [3],
                    layers: [
                    {
                        layer_type: "InputLayer",
                        layer: {
                              base: {
                                    layerBindingId: 0,
                                    base: {
                                        index: 0,
                                        layerName: "InputLayer1",
                                        layerType: "Input",
                                        inputSlots: [{
                                            index: 0,
                                            connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                        }],
                                        outputSlots: [ {
                                            index: 0,
                                            tensorInfo: {
                                                dimensions: )" + inputShape1 + R"(,
                                                dataType: )" + dataType + R"(
                                            },
                                        }],
                                     },}},
                    },
                    {
                    layer_type: "InputLayer",
                    layer: {
                           base: {
                                layerBindingId: 1,
                                base: {
                                      index:1,
                                      layerName: "InputLayer2",
                                      layerType: "Input",
                                      inputSlots: [{
                                          index: 0,
                                          connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                      }],
                                      outputSlots: [ {
                                          index: 0,
                                          tensorInfo: {
                                              dimensions: )" + inputShape2 + R"(,
                                              dataType: )" + dataType + R"(
                                          },
                                      }],
                                    },}},
                    },
                    {
                        layer_type: "ElementwiseBinaryLayer",
                        layer: {
                            base: {
                                index: 2,
                                layerName: "ElementwiseBinaryLayer",
                                layerType: "ElementwiseBinary",
                                inputSlots: [                                            {
                                             index: 0,
                                             connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                            },
                                            {
                                             index: 1,
                                             connection: {sourceLayerIndex:1, outputSlotIndex:0 },
                                            }],
                                outputSlots: [{
                                    index: 0,
                                    tensorInfo: {
                                        dimensions: )" + outputShape + R"(,
                                        dataType: )" + dataType + R"(
                                    }
                                }]
                            },
                            descriptor: {
                                operation: )" + binaryOperation + R"(
                            },
                        }
                    },
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
                                    outputSlots: [{
                                        index: 0,
                                        tensorInfo: {
                                            dimensions: )" + outputShape + R"(,
                                            dataType: )" + dataType + R"(
                                        },
                                    }],
                                }
                            }
                        },
                    }
                ]
            }
        )";
            Setup();
        }
    };

struct SimplePowerFixture : ElementwiseBinaryFixture
{
    SimplePowerFixture() : ElementwiseBinaryFixture("[ 2, 2 ]",
                                                    "[ 2, 2 ]",
                                                    "[ 2, 2 ]",
                                                    "QuantisedAsymm8",
                                                    "Power") {}
};

TEST_CASE_FIXTURE(SimplePowerFixture, "PowerQuantisedAsymm8")
{
    RunTest<2, armnn::DataType::QAsymmU8>(
            0,
            {{"InputLayer1", { 0, 1, 2, 3 }},
             {"InputLayer2", { 4, 5, 3, 2 }}},
            {{"OutputLayer", { 0, 1, 8, 9 }}});
}

struct SimpleSquaredDifferenceFixture : ElementwiseBinaryFixture
{
    SimpleSquaredDifferenceFixture() : ElementwiseBinaryFixture("[ 2, 2 ]",
                                                                "[ 2, 2 ]",
                                                                "[ 2, 2 ]",
                                                                "QuantisedAsymm8",
                                                                "SqDiff") {}
};

TEST_CASE_FIXTURE(SimpleSquaredDifferenceFixture, "SquaredDifferenceQuantisedAsymm8")
{
    RunTest<2, armnn::DataType::QAsymmU8>(
            0,
            {{"InputLayer1", { 5, 1, 7, 9 }},
             {"InputLayer2", { 4, 5, 2, 1 }}},
            {{"OutputLayer", { 1, 16, 25, 64 }}});
}

}