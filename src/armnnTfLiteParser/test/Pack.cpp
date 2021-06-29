//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Pack")
{
struct PackFixture : public ParserFlatbuffersFixture
{
    explicit PackFixture(const std::string & inputShape,
                         const unsigned int numInputs,
                         const std::string & outputShape,
                         const std::string & axis)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "PACK" } ],
                "subgraphs": [ {
                    "tensors": [)";

        for (unsigned int i = 0; i < numInputs; ++i)
        {
            m_JsonString += R"(
                        {
                            "shape": )" + inputShape + R"(,
                            "type": "FLOAT32",
                            "buffer": )" + std::to_string(i) + R"(,
                            "name": "inputTensor)" + std::to_string(i + 1) + R"(",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },)";
        }

        std::string inputIndexes;
        for (unsigned int i = 0; i < numInputs-1; ++i)
        {
            inputIndexes += std::to_string(i) + R"(, )";
        }
        inputIndexes += std::to_string(numInputs-1);

        m_JsonString += R"(
                        {
                            "shape": )" + outputShape + R"( ,
                            "type": "FLOAT32",
                            "buffer": )" + std::to_string(numInputs) + R"(,
                            "name": "outputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        }
                    ],
                    "inputs": [ )" + inputIndexes + R"( ],
                    "outputs": [ 2 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ )" + inputIndexes + R"( ],
                            "outputs": [ 2 ],
                            "builtin_options_type": "PackOptions",
                            "builtin_options": {
                                "axis": )" + axis + R"(,
                                "values_count": )" + std::to_string(numInputs) + R"(
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [)";
            
            for (unsigned int i = 0; i < numInputs-1; ++i)
            {
                m_JsonString += R"(
                    { },)";
            }
            m_JsonString += R"(
                    { }
                ]
            })";
        Setup();
    }
};

struct SimplePackFixture : PackFixture
{
    SimplePackFixture() : PackFixture("[ 3, 2, 3 ]",
                                      2,
                                      "[ 3, 2, 3, 2 ]",
                                      "3") {}
};

TEST_CASE_FIXTURE(SimplePackFixture, "ParsePack")
{
    RunTest<4, armnn::DataType::Float32>(
    0,
    { {"inputTensor1", { 1, 2, 3,
                         4, 5, 6,

                         7, 8, 9,
                         10, 11, 12,

                         13, 14, 15,
                         16, 17, 18 } },
    {"inputTensor2", { 19, 20, 21,
                       22, 23, 24,

                       25, 26, 27,
                       28, 29, 30,

                       31, 32, 33,
                       34, 35, 36 } } },
    { {"outputTensor", { 1, 19,
                         2, 20,
                         3, 21,

                         4, 22,
                         5, 23,
                         6, 24,


                         7, 25,
                         8, 26,
                         9, 27,

                         10, 28,
                         11, 29,
                         12, 30,
 

                         13, 31,
                         14, 32,
                         15, 33,

                         16, 34,
                         17, 35,
                         18, 36 } } });
}

}
