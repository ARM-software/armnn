//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Unpack")
{
struct UnpackFixture : public ParserFlatbuffersFixture
{
    explicit UnpackFixture(const std::string& inputShape,
                           const unsigned int numberOfOutputs,
                           const std::string& outputShape,
                           const std::string& axis,
                           const std::string& num,
                           const std::string& dataType,
                           const std::string& outputScale,
                           const std::string& outputOffset)
    {
        // As input index is 0, output indexes start at 1
        std::string outputIndexes = "1";
        for(unsigned int i = 1; i < numberOfOutputs; i++)
        {
            outputIndexes += ", " + std::to_string(i+1);
        }
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "UNPACK" } ],
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
                        },)";
        // Append the required number of outputs for this UnpackFixture.
        // As input index is 0, output indexes start at 1.
        for(unsigned int i = 0; i < numberOfOutputs; i++)
        {
            m_JsonString += R"(
                        {
                            "shape": )" + outputShape + R"( ,
                                "type": )" + dataType + R"(,
                                "buffer": )" + std::to_string(i + 1) + R"(,
                                "name": "outputTensor)" + std::to_string(i + 1) + R"(",
                                "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ )" + outputScale + R"( ],
                                "zero_point": [ )" + outputOffset + R"( ],
                            }
                        },)";
        }
        m_JsonString += R"(
                    ],
                    "inputs": [ 0 ],
                    "outputs": [ )" + outputIndexes + R"( ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0 ],
                            "outputs": [ )" + outputIndexes + R"( ],
                            "builtin_options_type": "UnpackOptions",
                            "builtin_options": {
                                "axis": )" + axis;

                    if(!num.empty())
                    {
                        m_JsonString += R"(,
                                "num" : )" + num;
                    }

                    m_JsonString += R"(
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { }
                ]
            }
        )";
        Setup();
    }
};

struct DefaultUnpackAxisZeroFixture : UnpackFixture
{
    DefaultUnpackAxisZeroFixture() : UnpackFixture("[ 4, 1, 6 ]", 4, "[ 1, 6 ]", "0", "", "FLOAT32", "1.0", "0") {}
};

struct DefaultUnpackAxisZeroUint8Fixture : UnpackFixture
{
    DefaultUnpackAxisZeroUint8Fixture() : UnpackFixture("[ 4, 1, 6 ]", 4, "[ 1, 6 ]", "0", "", "UINT8", "0.1", "0") {}
};

TEST_CASE_FIXTURE(DefaultUnpackAxisZeroFixture, "UnpackAxisZeroNumIsDefaultNotSpecified")
{
    RunTest<2, armnn::DataType::Float32>(
        0,
        { {"inputTensor", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                            13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                            19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f } } },
        { {"outputTensor1", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }},
          {"outputTensor2", { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f }},
          {"outputTensor3", { 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f }},
          {"outputTensor4", { 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f }} });
}

TEST_CASE_FIXTURE(DefaultUnpackAxisZeroUint8Fixture, "UnpackAxisZeroNumIsDefaultNotSpecifiedUint8")
{
    RunTest<2, armnn::DataType::QAsymmU8>(
        0,
        { {"inputTensor", { 1, 2, 3, 4, 5, 6,
                            7, 8, 9, 10, 11, 12,
                            13, 14, 15, 16, 17, 18,
                            19, 20, 21, 22, 23, 24 } } },
        { {"outputTensor1", { 10, 20, 30, 40, 50, 60 }},
          {"outputTensor2", { 70, 80, 90, 100, 110, 120 }},
          {"outputTensor3", { 130, 140, 150, 160, 170, 180 }},
          {"outputTensor4", { 190, 200, 210, 220, 230, 240 }} });
}

struct DefaultUnpackLastAxisFixture : UnpackFixture
{
    DefaultUnpackLastAxisFixture() : UnpackFixture("[ 4, 1, 6 ]", 6, "[ 4, 1 ]", "2", "6", "FLOAT32", "1.0", "0") {}
};

struct DefaultUnpackLastAxisUint8Fixture : UnpackFixture
{
    DefaultUnpackLastAxisUint8Fixture() : UnpackFixture("[ 4, 1, 6 ]", 6, "[ 4, 1 ]", "2", "6", "UINT8", "0.1", "0") {}
};

TEST_CASE_FIXTURE(DefaultUnpackLastAxisFixture, "UnpackLastAxisNumSix")
{
    RunTest<2, armnn::DataType::Float32>(
        0,
        { {"inputTensor", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                            13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                            19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f } } },
        { {"outputTensor1", { 1.0f, 7.0f, 13.0f, 19.0f }},
          {"outputTensor2", { 2.0f, 8.0f, 14.0f, 20.0f }},
          {"outputTensor3", { 3.0f, 9.0f, 15.0f, 21.0f }},
          {"outputTensor4", { 4.0f, 10.0f, 16.0f, 22.0f }},
          {"outputTensor5", { 5.0f, 11.0f, 17.0f, 23.0f }},
          {"outputTensor6", { 6.0f, 12.0f, 18.0f, 24.0f }} });
}

TEST_CASE_FIXTURE(DefaultUnpackLastAxisUint8Fixture, "UnpackLastAxisNumSixUint8") {
    RunTest<2, armnn::DataType::QAsymmU8>(
        0,
        {{"inputTensor", { 1, 2, 3, 4, 5, 6,
                           7, 8, 9, 10, 11, 12,
                           13, 14, 15, 16, 17, 18,
                           19, 20, 21, 22, 23, 24 }}},
        {{"outputTensor1", { 10, 70, 130, 190 }},
         {"outputTensor2", { 20, 80, 140, 200 }},
         {"outputTensor3", { 30, 90, 150, 210 }},
         {"outputTensor4", { 40, 100, 160, 220 }},
         {"outputTensor5", { 50, 110, 170, 230 }},
         {"outputTensor6", { 60, 120, 180, 240 }}});
}

}
