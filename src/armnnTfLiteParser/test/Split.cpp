//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"

#include <string>
#include <iostream>

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser)

struct SplitFixture : public ParserFlatbuffersFixture
{
    explicit SplitFixture(const std::string& inputShape,
                          const std::string& axisShape,
                          const std::string& numSplits,
                          const std::string& outputShape1,
                          const std::string& outputShape2,
                          const std::string& axisData)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "SPLIT" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + inputShape + R"(,
                            "type": "FLOAT32",
                            "buffer": 0,
                            "name": "inputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": )" + axisShape + R"(,
                            "type": "INT32",
                            "buffer": 1,
                            "name": "axis",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": )" + outputShape1 + R"( ,
                            "type": "FLOAT32",
                            "buffer": 2,
                            "name": "outputTensor1",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": )" + outputShape2 + R"( ,
                            "type": "FLOAT32",
                            "buffer": 3,
                            "name": "outputTensor2",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        }
                    ],
                    "inputs": [ 0 ],
                    "outputs": [ 2, 3 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 1, 0 ],
                            "outputs": [ 2, 3 ],
                            "builtin_options_type": "SplitOptions",
                            "builtin_options": {
                                "num_splits": )" + numSplits + R"(
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [ {}, {"data": )" + axisData + R"( }, {}, {} ]
            }
        )";

        Setup();
    }
};


struct SimpleSplitFixture : SplitFixture
{
    SimpleSplitFixture() : SplitFixture( "[ 2, 2, 2, 2 ]", "[ ]", "2",
        "[ 2, 1, 2, 2 ]", "[ 2, 1, 2, 2 ]", "[ 1, 0, 0, 0 ]")
         {}
};

BOOST_FIXTURE_TEST_CASE(ParseAxisOneSplitTwo, SimpleSplitFixture)
{

    RunTest<4, armnn::DataType::Float32>(
        0,
        { {"inputTensor", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                            11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f } } },
        { {"outputTensor1", { 1.0f, 2.0f, 3.0f, 4.0f, 9.0f, 10.0f, 11.0f, 12.0f } },
          {"outputTensor2", { 5.0f, 6.0f, 7.0f, 8.0f, 13.0f, 14.0f, 15.0f, 16.0f } } });
}

struct SimpleSplitAxisThreeFixture : SplitFixture
{
    SimpleSplitAxisThreeFixture() : SplitFixture( "[ 2, 2, 2, 2 ]", "[ ]", "2",
        "[ 2, 2, 2, 1 ]", "[ 2, 2, 2, 1 ]", "[ 3, 0, 0, 0 ]")
    {}
};

BOOST_FIXTURE_TEST_CASE(ParseAxisThreeSplitTwo, SimpleSplitAxisThreeFixture)
{
    RunTest<4, armnn::DataType::Float32>(
        0,
        { {"inputTensor", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                            11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f } } },
        { {"outputTensor1", { 1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f, 15.0f } },
          {"outputTensor2", { 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f } } } );
}

struct SimpleSplit2DFixture : SplitFixture
{
    SimpleSplit2DFixture() : SplitFixture( "[ 1, 8 ]", "[ ]", "2", "[ 1, 4 ]", "[ 1, 4 ]", "[ 1, 0, 0, 0 ]")
    {}
};

BOOST_FIXTURE_TEST_CASE(SimpleSplit2D, SimpleSplit2DFixture)
{
    RunTest<2, armnn::DataType::Float32>(
            0,
            { {"inputTensor", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f } } },
            { {"outputTensor1", { 1.0f, 2.0f, 3.0f, 4.0f } },
              {"outputTensor2", { 5.0f, 6.0f, 7.0f, 8.0f } } } );
}

struct SimpleSplit3DFixture : SplitFixture
{
    SimpleSplit3DFixture() : SplitFixture( "[ 1, 8, 2 ]", "[ ]", "2", "[ 1, 4, 2 ]", "[ 1, 4, 2 ]", "[ 1, 0, 0, 0 ]")
    {}
};

BOOST_FIXTURE_TEST_CASE(SimpleSplit3D, SimpleSplit3DFixture)
{
    RunTest<3, armnn::DataType::Float32>(
            0,
            { {"inputTensor", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                                10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f } } },
            { {"outputTensor1", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f } },
              {"outputTensor2", { 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f } } } );
}

BOOST_AUTO_TEST_SUITE_END()