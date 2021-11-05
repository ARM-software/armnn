//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_Pooling3d")
{
struct Pooling3dFixture : public ParserFlatbuffersSerializeFixture
{
    explicit Pooling3dFixture(const std::string &inputShape,
                              const std::string &outputShape,
                              const std::string &dataType,
                              const std::string &dataLayout,
                              const std::string &poolingAlgorithm)
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
                layer_type: "Pooling3dLayer",
                layer: {
                      base: {
                           index: 1,
                           layerName: "Pooling3dLayer",
                           layerType: "Pooling3d",
                           inputSlots: [{
                                  index: 0,
                                  connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                           }],
                           outputSlots: [ {
                                  index: 0,
                                  tensorInfo: {
                                       dimensions: )" + outputShape + R"(,
                                       dataType: )" + dataType + R"(

                           }}]},
                      descriptor: {
                           poolType: )" + poolingAlgorithm + R"(,
                           outputShapeRounding: "Floor",
                           paddingMethod: Exclude,
                           dataLayout: )" + dataLayout + R"(,
                           padLeft: 0,
                           padRight: 0,
                           padTop: 0,
                           padBottom: 0,
                           padFront: 0,
                           padBack: 0,
                           poolWidth: 2,
                           poolHeight: 2,
                           poolDepth: 2,
                           strideX: 2,
                           strideY: 2,
                           strideZ: 2
                           }
                }},
                {
                layer_type: "OutputLayer",
                layer: {
                    base:{
                          layerBindingId: 0,
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

struct SimpleAvgPooling3dFixture : Pooling3dFixture
{
    SimpleAvgPooling3dFixture() : Pooling3dFixture("[ 1, 2, 2, 2, 1 ]",
                                                   "[ 1, 1, 1, 1, 1 ]",
                                                   "Float32", "NDHWC", "Average") {}
};

struct SimpleAvgPooling3dFixture2 : Pooling3dFixture
{
    SimpleAvgPooling3dFixture2() : Pooling3dFixture("[ 1, 2, 2, 2, 1 ]",
                                                    "[ 1, 1, 1, 1, 1 ]",
                                                    "QuantisedAsymm8", "NDHWC", "Average") {}
};

struct SimpleMaxPooling3dFixture : Pooling3dFixture
{
    SimpleMaxPooling3dFixture() : Pooling3dFixture("[ 1, 1, 2, 2, 2 ]",
                                                   "[ 1, 1, 1, 1, 1 ]",
                                                   "Float32", "NCDHW", "Max") {}
};

struct SimpleMaxPooling3dFixture2 : Pooling3dFixture
{
    SimpleMaxPooling3dFixture2() : Pooling3dFixture("[ 1, 1, 2, 2, 2 ]",
                                                    "[ 1, 1, 1, 1, 1 ]",
                                                    "QuantisedAsymm8", "NCDHW", "Max") {}
};

struct SimpleL2Pooling3dFixture : Pooling3dFixture
{
    SimpleL2Pooling3dFixture() : Pooling3dFixture("[ 1, 2, 2, 2, 1 ]",
                                                  "[ 1, 1, 1, 1, 1 ]",
                                                  "Float32", "NDHWC", "L2") {}
};

TEST_CASE_FIXTURE(SimpleAvgPooling3dFixture, "Pooling3dFloat32Avg")
{
    RunTest<5, armnn::DataType::Float32>(0, { 2, 3, 5, 2, 3, 2, 3, 4 }, { 3 });
}

TEST_CASE_FIXTURE(SimpleAvgPooling3dFixture2, "Pooling3dQuantisedAsymm8Avg")
{
    RunTest<5, armnn::DataType::QAsymmU8>(0,{ 20, 40, 60, 80, 50, 60, 70, 30 },{ 50 });
}

TEST_CASE_FIXTURE(SimpleMaxPooling3dFixture, "Pooling3dFloat32Max")
{
    RunTest<5, armnn::DataType::Float32>(0, { 2, 5, 5, 2, 1, 3, 4, 0 }, { 5 });
}

TEST_CASE_FIXTURE(SimpleMaxPooling3dFixture2, "Pooling3dQuantisedAsymm8Max")
{
    RunTest<5, armnn::DataType::QAsymmU8>(0,{ 20, 40, 60, 80, 10, 40, 0, 70 },{ 80 });
}

TEST_CASE_FIXTURE(SimpleL2Pooling3dFixture, "Pooling3dFloat32L2")
{
    RunTest<5, armnn::DataType::Float32>(0, { 2, 3, 5, 2, 4, 1, 1, 3 }, { 2.93683503112f });
}

}

