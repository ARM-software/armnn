//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser")
{
struct SplitVFixture : public ParserFlatbuffersFixture
{
    explicit SplitVFixture(const std::string& inputShape,
                           const std::string& splitValues,
                           const std::string& sizeSplitsShape,
                           const std::string& axisShape,
                           const std::string& numSplits,
                           const std::string& outputShape1,
                           const std::string& outputShape2,
                           const std::string& axisData,
                           const std::string& dataType)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "SPLIT_V" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + inputShape + R"(,
                            "type": )" + dataType + R"(,
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
                            "shape": )" + sizeSplitsShape + R"(,
                            "type": "INT32",
                            "buffer": 1,
                            "name": "sizeSplits",
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
                            "buffer": 2,
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
                            "type":)" + dataType + R"(,
                            "buffer": 3,
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
                            "type":)" + dataType + R"(,
                            "buffer": 4,
                            "name": "outputTensor2",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        }
                    ],
                    "inputs": [ 0, 1, 2 ],
                    "outputs": [ 3, 4 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0, 1, 2 ],
                            "outputs": [ 3, 4 ],
                            "builtin_options_type": "SplitVOptions",
                            "builtin_options": {
                                "num_splits": )" + numSplits + R"(
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [ {}, { "data": )" + splitValues + R"( }, { "data": )" + axisData + R"( }, {}, {}]
            }
        )";

        Setup();
    }
};

/*
 *  Tested inferred splitSizes with splitValues [-1, 1] locally.
 */

struct SimpleSplitVAxisOneFixture : SplitVFixture
{
    SimpleSplitVAxisOneFixture()
        : SplitVFixture( "[ 4, 2, 2, 2 ]", "[ 1, 0, 0, 0, 3, 0, 0, 0 ]", "[ 2 ]","[ ]", "2",
                         "[ 1, 2, 2, 2 ]", "[ 3, 2, 2, 2 ]", "[ 0, 0, 0, 0 ]", "FLOAT32")
    {}
};

TEST_CASE_FIXTURE(SimpleSplitVAxisOneFixture, "ParseAxisOneSplitVTwo")
{
    RunTest<4, armnn::DataType::Float32>(
        0,
        { {"inputTensor",   { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                              17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                              25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f } } },
        { {"outputTensor1", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f } },
          {"outputTensor2", { 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                              17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                              25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f } } } );
}

struct SimpleSplitVAxisTwoFixture : SplitVFixture
{
    SimpleSplitVAxisTwoFixture()
        : SplitVFixture( "[ 2, 4, 2, 2 ]", "[ 3, 0, 0, 0, 1, 0, 0, 0 ]", "[ 2 ]","[ ]", "2",
                         "[ 2, 3, 2, 2 ]", "[ 2, 1, 2, 2 ]", "[ 1, 0, 0, 0 ]", "FLOAT32")
    {}
};

TEST_CASE_FIXTURE(SimpleSplitVAxisTwoFixture, "ParseAxisTwoSplitVTwo")
{
    RunTest<4, armnn::DataType::Float32>(
        0,
        { {"inputTensor",   { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                              17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                              25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f } } },
        { {"outputTensor1", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                              21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f } },
          {"outputTensor2", { 13.0f, 14.0f, 15.0f, 16.0f, 29.0f, 30.0f, 31.0f, 32.0f } } } );
}

struct SimpleSplitVAxisThreeFixture : SplitVFixture
{
    SimpleSplitVAxisThreeFixture()
        : SplitVFixture( "[ 2, 2, 4, 2 ]", "[ 1, 0, 0, 0, 3, 0, 0, 0 ]", "[ 2 ]","[ ]", "2",
                         "[ 2, 2, 1, 2 ]", "[ 2, 2, 3, 2 ]", "[ 2, 0, 0, 0 ]", "FLOAT32")
    {}
};

TEST_CASE_FIXTURE(SimpleSplitVAxisThreeFixture, "ParseAxisThreeSplitVTwo")
{
    RunTest<4, armnn::DataType::Float32>(
        0,
        { {"inputTensor",   { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                              17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                              25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f } } },
        { {"outputTensor1", { 1.0f, 2.0f, 9.0f, 10.0f, 17.0f, 18.0f, 25.0f, 26.0f } },
          {"outputTensor2", { 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 11.0f, 12.0f,
                              13.0f, 14.0f, 15.0f, 16.0f, 19.0f, 20.0f, 21.0f, 22.0f,
                              23.0f, 24.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f } } } );
}

struct SimpleSplitVAxisFourFixture : SplitVFixture
{
    SimpleSplitVAxisFourFixture()
        : SplitVFixture( "[ 2, 2, 2, 4 ]", "[ 3, 0, 0, 0, 1, 0, 0, 0 ]", "[ 2 ]","[ ]", "2",
                         "[ 2, 2, 2, 3 ]", "[ 2, 2, 2, 1 ]", "[ 3, 0, 0, 0 ]", "FLOAT32")
    {}
};

TEST_CASE_FIXTURE(SimpleSplitVAxisFourFixture, "ParseAxisFourSplitVTwo")
{
    RunTest<4, armnn::DataType::Float32>(
        0,
        { {"inputTensor",   { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                              17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                              25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f } } },
        { {"outputTensor1", { 1.0f, 2.0f, 3.0f, 5.0f, 6.0f, 7.0f, 9.0f, 10.0f,
                              11.0f, 13.0f, 14.0f, 15.0f, 17.0f, 18.0f, 19.0f, 21.0f,
                              22.0f, 23.0f, 25.0f, 26.0f, 27.0f, 29.0f, 30.0f, 31.0f} },
          {"outputTensor2", { 4.0f, 8.0f, 12.0f, 16.0f, 20.0f, 24.0f, 28.0f, 32.0f } } } );
}

}
