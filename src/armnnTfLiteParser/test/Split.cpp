//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Split")
{
struct SplitFixture : public ParserFlatbuffersFixture
{
    explicit SplitFixture(const std::string& inputShape,
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
                "operator_codes": [ { "builtin_code": "SPLIT" } ],
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
                            "type":)" + dataType + R"(,
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
                            "type":)" + dataType + R"(,
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


struct SimpleSplitFixtureFloat32 : SplitFixture
{
    SimpleSplitFixtureFloat32()
        : SplitFixture( "[ 2, 2, 2, 2 ]", "[ ]", "2", "[ 2, 1, 2, 2 ]", "[ 2, 1, 2, 2 ]", "[ 1, 0, 0, 0 ]", "FLOAT32")
        {}
};

TEST_CASE_FIXTURE(SimpleSplitFixtureFloat32, "ParseAxisOneSplitTwoFloat32")
{

    RunTest<4, armnn::DataType::Float32>(
        0,
        { {"inputTensor", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f } } },
        { {"outputTensor1", { 1.0f, 2.0f, 3.0f, 4.0f, 9.0f, 10.0f, 11.0f, 12.0f } },
          {"outputTensor2", { 5.0f, 6.0f, 7.0f, 8.0f, 13.0f, 14.0f, 15.0f, 16.0f } } });
}

struct SimpleSplitAxisThreeFixtureFloat32 : SplitFixture
{
    SimpleSplitAxisThreeFixtureFloat32()
        : SplitFixture( "[ 2, 2, 2, 2 ]", "[ ]", "2", "[ 2, 2, 2, 1 ]", "[ 2, 2, 2, 1 ]", "[ 3, 0, 0, 0 ]", "FLOAT32")
        {}
};

TEST_CASE_FIXTURE(SimpleSplitAxisThreeFixtureFloat32, "ParseAxisThreeSplitTwoFloat32")
{
    RunTest<4, armnn::DataType::Float32>(
        0,
        { {"inputTensor", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f } } },
        { {"outputTensor1", { 1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f, 15.0f } },
          {"outputTensor2", { 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f } } } );
}

struct SimpleSplit2DFixtureFloat32 : SplitFixture
{
    SimpleSplit2DFixtureFloat32()
        : SplitFixture( "[ 1, 8 ]", "[ ]", "2", "[ 1, 4 ]", "[ 1, 4 ]", "[ 1, 0, 0, 0 ]", "FLOAT32")
        {}
};

TEST_CASE_FIXTURE(SimpleSplit2DFixtureFloat32, "SimpleSplit2DFloat32")
{
    RunTest<2, armnn::DataType::Float32>(
        0,
        { {"inputTensor", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f } } },
        { {"outputTensor1", { 1.0f, 2.0f, 3.0f, 4.0f } },
          {"outputTensor2", { 5.0f, 6.0f, 7.0f, 8.0f } } } );
}

struct SimpleSplit3DFixtureFloat32 : SplitFixture
{
    SimpleSplit3DFixtureFloat32()
        : SplitFixture( "[ 1, 8, 2 ]", "[ ]", "2", "[ 1, 4, 2 ]", "[ 1, 4, 2 ]", "[ 1, 0, 0, 0 ]", "FLOAT32")
        {}
};

TEST_CASE_FIXTURE(SimpleSplit3DFixtureFloat32, "SimpleSplit3DFloat32")
{
    RunTest<3, armnn::DataType::Float32>(
        0,
        { {"inputTensor", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f } } },
        { {"outputTensor1", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f } },
          {"outputTensor2", { 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f } } } );
}

struct SimpleSplitFixtureUint8 : SplitFixture
{
    SimpleSplitFixtureUint8()
        : SplitFixture( "[ 2, 2, 2, 2 ]", "[ ]", "2", "[ 2, 1, 2, 2 ]", "[ 2, 1, 2, 2 ]", "[ 1, 0, 0, 0 ]", "UINT8")
        {}
};

TEST_CASE_FIXTURE(SimpleSplitFixtureUint8, "ParseAxisOneSplitTwoUint8")
{

    RunTest<4, armnn::DataType::QAsymmU8>(
        0,
        { {"inputTensor", { 1, 2, 3, 4, 5, 6, 7, 8,
                            9, 10, 11, 12, 13, 14, 15, 16 } } },
        { {"outputTensor1", { 1, 2, 3, 4, 9, 10, 11, 12 } },
          {"outputTensor2", { 5, 6, 7, 8, 13, 14, 15, 16 } } });
}

struct SimpleSplitAxisThreeFixtureUint8 : SplitFixture
{
    SimpleSplitAxisThreeFixtureUint8()
        : SplitFixture( "[ 2, 2, 2, 2 ]", "[ ]", "2", "[ 2, 2, 2, 1 ]", "[ 2, 2, 2, 1 ]", "[ 3, 0, 0, 0 ]", "UINT8")
        {}
};

TEST_CASE_FIXTURE(SimpleSplitAxisThreeFixtureUint8, "ParseAxisThreeSplitTwoUint8")
{
    RunTest<4, armnn::DataType::QAsymmU8>(
        0,
        { {"inputTensor", { 1, 2, 3, 4, 5, 6, 7, 8,
                            9, 10, 11, 12, 13, 14, 15, 16 } } },
        { {"outputTensor1", { 1, 3, 5, 7, 9, 11, 13, 15 } },
          {"outputTensor2", { 2, 4, 6, 8, 10, 12, 14, 16 } } } );
}

struct SimpleSplit2DFixtureUint8 : SplitFixture
{
    SimpleSplit2DFixtureUint8()
        : SplitFixture( "[ 1, 8 ]", "[ ]", "2", "[ 1, 4 ]", "[ 1, 4 ]", "[ 1, 0, 0, 0 ]", "UINT8")
        {}
};

TEST_CASE_FIXTURE(SimpleSplit2DFixtureUint8, "SimpleSplit2DUint8")
{
    RunTest<2, armnn::DataType::QAsymmU8>(
            0,
            { {"inputTensor", { 1, 2, 3, 4, 5, 6, 7, 8 } } },
            { {"outputTensor1", { 1, 2, 3, 4 } },
              {"outputTensor2", { 5, 6, 7, 8 } } } );
}

struct SimpleSplit3DFixtureUint8 : SplitFixture
{
    SimpleSplit3DFixtureUint8()
        : SplitFixture( "[ 1, 8, 2 ]", "[ ]", "2", "[ 1, 4, 2 ]", "[ 1, 4, 2 ]", "[ 1, 0, 0, 0 ]", "UINT8")
        {}
};

TEST_CASE_FIXTURE(SimpleSplit3DFixtureUint8, "SimpleSplit3DUint8")
{
    RunTest<3, armnn::DataType::QAsymmU8>(
        0,
        { {"inputTensor", { 1, 2, 3, 4, 5, 6, 7, 8,
                            9, 10, 11, 12, 13, 14, 15, 16 } } },
        { {"outputTensor1", { 1, 2, 3, 4, 5, 6, 7, 8 } },
          {"outputTensor2", { 9, 10, 11, 12, 13, 14, 15, 16 } } } );
}

}