//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <QuantizeHelper.hpp>
#include <ResolveType.hpp>

#include <boost/test/unit_test.hpp>

#include <string>

BOOST_AUTO_TEST_SUITE(Deserializer)


struct CastFixture : public ParserFlatbuffersSerializeFixture
{
    explicit CastFixture(const std::string& inputShape,
                         const std::string& outputShape,
                         const std::string& inputDataType,
                         const std::string& outputDataType)
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
                                    layerName: "inputTensor",
                                    layerType: "Input",
                                    inputSlots: [{
                                        index: 0,
                                        connection: { sourceLayerIndex:0, outputSlotIndex:0 },
                                    }],
                                    outputSlots: [{
                                        index: 0,
                                        tensorInfo: {
                                            dimensions: )" + inputShape + R"(,
                                            dataType: )" + inputDataType + R"(
                                        }
                                    }]
                                }
                            }
                        }
                    },
                    {
                        layer_type: "CastLayer",
                        layer: {
                            base: {
                                 index:1,
                                 layerName: "CastLayer",
                                 layerType: "Cast",
                                 inputSlots: [{
                                     index: 0,
                                     connection: { sourceLayerIndex:0, outputSlotIndex:0 },
                                 }],
                                 outputSlots: [{
                                     index: 0,
                                     tensorInfo: {
                                         dimensions: )" + outputShape + R"(,
                                         dataType: )" + outputDataType + R"(
                                     },
                                 }],
                            },
                        },
                    },
                    {
                        layer_type: "OutputLayer",
                        layer: {
                            base:{
                                layerBindingId: 2,
                                base: {
                                    index: 2,
                                    layerName: "outputTensor",
                                    layerType: "Output",
                                    inputSlots: [{
                                        index: 0,
                                        connection: { sourceLayerIndex:1, outputSlotIndex:0 },
                                    }],
                                    outputSlots: [{
                                        index: 0,
                                        tensorInfo: {
                                            dimensions: )" + outputShape + R"(,
                                            dataType: )" + outputDataType + R"(
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

struct SimpleCastFixture : CastFixture
{
    SimpleCastFixture() : CastFixture("[ 1, 6 ]",
                                      "[ 1, 6 ]",
                                      "Signed32",
                                      "Float32") {}
};

BOOST_FIXTURE_TEST_CASE(SimpleCast, SimpleCastFixture)
{
RunTest<2, armnn::DataType::Signed32 , armnn::DataType::Float32>(
0,
{{"inputTensor",  { 0,   -1,   5,   -100,   200,   -255 }}},
{{"outputTensor", { 0.0f, -1.0f, 5.0f, -100.0f, 200.0f, -255.0f }}});
}

BOOST_AUTO_TEST_SUITE_END()
