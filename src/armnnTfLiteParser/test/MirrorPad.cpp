//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

TEST_SUITE("TensorflowLiteParser_MirrorPad")
{
struct MirrorPadFixture : public ParserFlatbuffersFixture
{
    explicit MirrorPadFixture(const std::string& inputShape,
                              const std::string& outputShape,
                              const std::string& padListShape,
                              const std::string& padListData,
                              const std::string& padMode,
                              const std::string& dataType = "FLOAT32",
                              const std::string& scale = "1.0",
                              const std::string& offset = "0")
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "MIRROR_PAD" } ],
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
                                "scale": [ )" + scale + R"( ],
                                "zero_point": [ )" + offset + R"( ],
                            }
                        },
                        {
                             "shape": )" + outputShape + R"(,
                             "type": )" + dataType + R"(,
                             "buffer": 1,
                             "name": "outputTensor",
                             "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ )" + scale + R"( ],
                                "zero_point": [ )" + offset + R"( ],
                            }
                        },
                        {
                             "shape": )" + padListShape + R"( ,
                             "type": "INT32",
                             "buffer": 2,
                             "name": "padList",
                             "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                             }
                        }
                    ],
                    "inputs": [ 0 ],
                    "outputs": [ 1 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0, 2 ],
                            "outputs": [ 1 ],
                            "builtin_options_type": "MirrorPadOptions",
                            "builtin_options": {
                              "mode": )" + padMode + R"( ,
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { },
                    { "data": )" + padListData + R"(, },
                ]
            }
        )";
      SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

struct SimpleMirrorPadSymmetricFixture : public MirrorPadFixture
{
    SimpleMirrorPadSymmetricFixture() : MirrorPadFixture("[ 3, 3 ]", "[ 7, 7 ]", "[ 2, 2 ]",
                                                         "[ 2,0,0,0, 2,0,0,0, 2,0,0,0, 2,0,0,0 ]",
                                                         "SYMMETRIC", "FLOAT32") {}
};

TEST_CASE_FIXTURE(SimpleMirrorPadSymmetricFixture, "ParseMirrorPadSymmetric")
{
    RunTest<2, armnn::DataType::Float32>
            (0,
             {{ "inputTensor",  { 1.0f, 2.0f, 3.0f,
                                  4.0f, 5.0f, 6.0f,
                                  7.0f, 8.0f, 9.0f }}},

             {{ "outputTensor", { 5.0f, 4.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f,
                                  2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                  2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                  5.0f, 4.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f,
                                  8.0f, 7.0f, 7.0f, 8.0f, 9.0f, 9.0f, 8.0f,
                                  8.0f, 7.0f, 7.0f, 8.0f, 9.0f, 9.0f, 8.0f,
                                  5.0f, 4.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f }}});
}

struct SimpleMirrorPadReflectFixture : public MirrorPadFixture
{
    SimpleMirrorPadReflectFixture() : MirrorPadFixture("[ 3, 3 ]", "[ 7, 7 ]", "[ 2, 2 ]",
                                                        "[ 2,0,0,0, 2,0,0,0, 2,0,0,0, 2,0,0,0 ]",
                                                        "REFLECT", "FLOAT32") {}
};

TEST_CASE_FIXTURE(SimpleMirrorPadReflectFixture, "ParseMirrorPadRelfect")
{
    RunTest<2, armnn::DataType::Float32>
        (0,
         {{ "inputTensor",  { 1.0f, 2.0f, 3.0f,
                              4.0f, 5.0f, 6.0f,
                              7.0f, 8.0f, 9.0f }}},

         {{ "outputTensor", { 9.0f, 8.0f, 7.0f, 8.0f, 9.0f, 8.0f, 7.0f,
                              6.0f, 5.0f, 4.0f, 5.0f, 6.0f, 5.0f, 4.0f,
                              3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f,
                              6.0f, 5.0f, 4.0f, 5.0f, 6.0f, 5.0f, 4.0f,
                              9.0f, 8.0f, 7.0f, 8.0f, 9.0f, 8.0f, 7.0f,
                              6.0f, 5.0f, 4.0f, 5.0f, 6.0f, 5.0f, 4.0f,
                              3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f }}});
}

}
