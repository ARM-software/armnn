//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("DeserializeParser_Constant")
{
struct ConstantAddFixture : public ParserFlatbuffersSerializeFixture
{
    explicit ConstantAddFixture(const std::string & shape,
                                const std::string & constTensorDatatype,
                                const std::string & constData,
                                const std::string & dataType)
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
                                    layerName: "InputLayer1",
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
                                        },
                                    }],
                                 },}},
                },
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
                                          dimensions: )" + shape + R"(,
                                          dataType: )" + dataType + R"(,
                                      },
                                  }],
                                  inputSlots: [{
                                        index: 0,
                                        connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                    }],
                                },
                            input: {
                                info: {
                                         dimensions: )" + shape + R"(,
                                         dataType: )" + dataType + R"(
                                     },
                                data_type: )" + constTensorDatatype + R"(,
                                data: {
                                    data: )" + constData + R"(,
                                    } }
                        },
                },
                {
                layer_type: "AdditionLayer",
                layer : {
                        base: {
                             index:2,
                             layerName: "AdditionLayer",
                             layerType: "Addition",
                             inputSlots: [
                                            {
                                             index: 0,
                                             connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                            },
                                            {
                                             index: 1,
                                             connection: {sourceLayerIndex:1, outputSlotIndex:0 },
                                            }
                             ],
                             outputSlots: [ {
                                 index: 0,
                                 tensorInfo: {
                                     dimensions: )" + shape + R"(,
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
                                            dimensions: )" + shape + R"(,
                                            dataType: )" + dataType + R"(
                                        },
                                }],
                            }}},
                }],
                featureVersions: {
                    weightsLayoutScheme: 1,
                }
         }
        )";
        SetupSingleInputSingleOutput("InputLayer1", "OutputLayer");
    }
};

struct SimpleConstantAddFixture : ConstantAddFixture
{
    SimpleConstantAddFixture()
            : ConstantAddFixture("[ 2, 3 ]",             // shape
                                 "ByteData",             // constDataType
                                 "[ 1, 2, 3, 4, 5, 6 ]", // constData
                                 "QuantisedAsymm8")      // datatype

    {}
};

TEST_CASE_FIXTURE(SimpleConstantAddFixture, "SimpleConstantAddQuantisedAsymm8")
{
    RunTest<2, armnn::DataType::QAsymmU8>(
            0,
            { 1, 2, 3, 4, 5, 6  },
            { 2, 4, 6, 8, 10, 12 });
}

}