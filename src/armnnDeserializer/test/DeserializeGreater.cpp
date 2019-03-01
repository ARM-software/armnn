//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersSerializeFixture.hpp"
#include "../Deserializer.hpp"

#include <string>
#include <iostream>

BOOST_AUTO_TEST_SUITE(Deserializer)

struct GreaterFixture : public ParserFlatbuffersSerializeFixture
{
    explicit GreaterFixture(const std::string & inputShape1,
                            const std::string & inputShape2,
                            const std::string & outputShape,
                            const std::string & inputDataType,
                            const std::string & outputDataType)
    {
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
                                            dataType: )" + inputDataType + R"(
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
                                          dataType: )" + inputDataType + R"(
                                      },
                                  }],
                                },}},
                },
                {
                layer_type: "GreaterLayer",
                layer : {
                        base: {
                             index:2,
                             layerName: "GreaterLayer",
                             layerType: "Greater",
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
                                     dimensions: )" + outputShape + R"(,
                                     dataType: Boolean
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
                                            dimensions: )" + outputShape + R"(,
                                            dataType: )" + outputDataType + R"(
                                        },
                                }],
                            }}},
                }]
         }
        )";
        Setup();
    }
};


struct SimpleGreaterFixtureQuantisedAsymm8 : GreaterFixture
{
    SimpleGreaterFixtureQuantisedAsymm8() : GreaterFixture("[ 2, 2 ]",        // input1Shape
                                                           "[ 2, 2 ]",        // input2Shape
                                                           "[ 2, 2 ]",        // outputShape
                                                           "QuantisedAsymm8", // inputDataType
                                                           "Float32") {}      // outputDataType
};

struct SimpleGreaterFixtureFloat32 : GreaterFixture
{
    SimpleGreaterFixtureFloat32() : GreaterFixture("[ 2, 2, 1, 1 ]", // input1Shape
                                                   "[ 2, 2, 1, 1 ]", // input2Shape
                                                   "[ 2, 2, 1, 1 ]", // outputShape
                                                   "Float32",        // inputDataType
                                                   "Float32") {}     // outputDataType
};

struct SimpleGreaterFixtureBroadcast : GreaterFixture
{
    SimpleGreaterFixtureBroadcast() : GreaterFixture("[ 1, 2, 2, 2 ]", // input1Shape
                                                     "[ 1, 1, 1, 1 ]", // input2Shape
                                                     "[ 1, 2, 2, 2 ]", // outputShape
                                                     "Float32",        // inputDataType
                                                     "Float32") {}     // outputDataType
};


BOOST_FIXTURE_TEST_CASE(GreaterQuantisedAsymm8, SimpleGreaterFixtureQuantisedAsymm8)
{
    RunTest<2, armnn::DataType::QuantisedAsymm8, armnn::DataType::Boolean>(
        0,
        {{"InputLayer1", { 1, 5, 8, 7 }},
        { "InputLayer2", { 4, 0, 6, 7 }}},
        {{"OutputLayer", { 0, 1, 1, 0 }}});
}

BOOST_FIXTURE_TEST_CASE(GreaterFloat32, SimpleGreaterFixtureFloat32)
{
    RunTest<4, armnn::DataType::Float32, armnn::DataType::Boolean>(
        0,
        {{"InputLayer1", { 1.0f, 2.0f, 3.0f, 4.0f }},
        { "InputLayer2", { 1.0f, 5.0f, 2.0f, 2.0f }}},
        {{"OutputLayer", { 0, 0, 1, 1 }}});
}

BOOST_FIXTURE_TEST_CASE(GreaterBroadcast, SimpleGreaterFixtureBroadcast)
{
    RunTest<4, armnn::DataType::Float32, armnn::DataType::Boolean>(
        0,
        {{"InputLayer1", { 1, 2, 3, 4, 5, 6, 7, 8 }},
         {"InputLayer2", { 1 }}},
        {{"OutputLayer", { 0, 1, 1, 1, 1, 1, 1, 1 }}});
}

BOOST_AUTO_TEST_SUITE_END()
