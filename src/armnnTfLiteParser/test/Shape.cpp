//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

TEST_SUITE("TensorflowLiteParser_Shape")
{
struct ShapeFixture : public ParserFlatbuffersFixture
{
    explicit ShapeFixture(const std::string& inputShape,
                          const std::string& outputShape,
                          const std::string& inputDataType,
                          const std::string& outputDataType)
     {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "SHAPE" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + inputShape + R"(,
                            "type": )" + inputDataType + R"(,
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
                            "shape": )" + outputShape + R"(,
                            "type": )" + outputDataType + R"(,
                            "buffer": 1,
                            "name": "outputTensor",
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
                          "inputs": [ 0 ],
                          "outputs": [ 1 ],
                          "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [ {}, {} ]
            }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};


struct SimpleShapeFixture : ShapeFixture
{
    SimpleShapeFixture() : ShapeFixture("[ 1, 3, 3, 1 ]",
                                       "[ 4 ]",
                                       "INT32",
                                       "INT32") {}
};

TEST_CASE_FIXTURE(SimpleShapeFixture, "SimpleShapeFixture")
{
    RunTest<1, armnn::DataType::Signed32>(
            0,
            {{"inputTensor", { 1, 1, 1, 1, 1, 1, 1, 1, 1 }}},
            {{"outputTensor",{ 1, 3, 3, 1 }}});
}

}