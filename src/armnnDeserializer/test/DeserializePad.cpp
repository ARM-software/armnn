//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_Pad")
{
struct PadFixture : public ParserFlatbuffersSerializeFixture
{
    explicit PadFixture(const std::string& inputShape,
                        const std::string& padList,
                        const std::string& outputShape,
                        const std::string& dataType,
                        const std::string& paddingMode)
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
                                    outputSlots: [{
                                        index: 0,
                                        tensorInfo: {
                                            dimensions: )" + inputShape + R"(,
                                            dataType: )" + dataType + R"(
                                        }
                                    }]
                                }
                            }
                        }
                    },
                    {
                        layer_type: "PadLayer",
                        layer: {
                            base: {
                                index: 1,
                                layerName: "PadLayer",
                                layerType: "Pad",
                                inputSlots: [{
                                    index: 0,
                                    connection: {sourceLayerIndex:0, outputSlotIndex:0 },
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
                                padList: )" + padList + R"(,
                                paddingMode: )" + paddingMode + R"(,
                            }
                        }
                    },
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
        SetupSingleInputSingleOutput("InputLayer", "OutputLayer");
    }
};

struct SimplePadFixture : PadFixture
{
    SimplePadFixture() : PadFixture("[ 2, 2, 2 ]",
                                    "[ 0, 1, 2, 1, 2, 2 ]",
                                    "[ 3, 5, 6 ]",
                                    "QuantisedAsymm8",
                                    "Constant") {}
};

TEST_CASE_FIXTURE(SimplePadFixture, "SimplePadQuantisedAsymm8")
{
    RunTest<3, armnn::DataType::QAsymmU8>(0,
                                         {
                                            0, 4, 2, 5, 6, 1, 5, 2
                                         },
                                         {
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            4, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,
                                            1, 0, 0, 0, 0, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                         });
}

struct SimplePadSymmetricFixture : PadFixture
{
    SimplePadSymmetricFixture() : PadFixture("[ 2, 2, 2 ]",
                                             "[ 1, 1, 1, 1, 1, 1 ]",
                                             "[ 4, 4, 4 ]",
                                             "QuantisedAsymm8",
                                             "Symmetric") {}
};

TEST_CASE_FIXTURE(SimplePadSymmetricFixture, "SimplePadSymmetricQuantisedAsymm8")
{
    RunTest<3, armnn::DataType::QAsymmU8>(0,
                                          {
                                              1, 2,
                                              3, 4,

                                              5, 6,
                                              7, 8
                                          },
                                          {
                                              1, 1, 2, 2,
                                              1, 1, 2, 2,
                                              3, 3, 4, 4,
                                              3, 3, 4, 4,

                                              1, 1, 2, 2,
                                              1, 1, 2, 2,
                                              3, 3, 4, 4,
                                              3, 3, 4, 4,

                                              5, 5, 6, 6,
                                              5, 5, 6, 6,
                                              7, 7, 8, 8,
                                              7, 7, 8, 8,

                                              5, 5, 6, 6,
                                              5, 5, 6, 6,
                                              7, 7, 8, 8,
                                              7, 7, 8, 8
                                          });
}

struct SimplePadReflectFixture : PadFixture
{
    SimplePadReflectFixture() : PadFixture("[ 2, 2, 2 ]",
                                           "[ 1, 1, 1, 1, 1, 1 ]",
                                           "[ 4, 4, 4 ]",
                                           "QuantisedAsymm8",
                                           "Reflect") {}
};

TEST_CASE_FIXTURE(SimplePadReflectFixture, "SimplePadReflectQuantisedAsymm8")
{
    RunTest<3, armnn::DataType::QAsymmU8>(0,
                                          {
                                              1, 2,
                                              3, 4,

                                              5, 6,
                                              7, 8
                                          },
                                          {
                                              8, 7, 8, 7,
                                              6, 5, 6, 5,
                                              8, 7, 8, 7,
                                              6, 5, 6, 5,

                                              4, 3, 4, 3,
                                              2, 1, 2, 1,
                                              4, 3, 4, 3,
                                              2, 1, 2, 1,

                                              8, 7, 8, 7,
                                              6, 5, 6, 5,
                                              8, 7, 8, 7,
                                              6, 5, 6, 5,

                                              4, 3, 4, 3,
                                              2, 1, 2, 1,
                                              4, 3, 4, 3,
                                              2, 1, 2, 1
                                          });
}

}
