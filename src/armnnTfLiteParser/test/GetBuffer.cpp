//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"
#include <sstream>

using armnnTfLiteParser::TfLiteParser;

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser)

struct GetBufferFixture : public ParserFlatbuffersFixture
{
    explicit GetBufferFixture()
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "CONV_2D" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": [ 1, 3, 3, 1 ],
                            "type": "UINT8",
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
                            "shape": [ 1, 1, 1, 1 ],
                            "type": "UINT8",
                            "buffer": 1,
                            "name": "outputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 511.0 ],
                                "scale": [ 2.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": [ 1, 3, 3, 1 ],
                            "type": "UINT8",
                            "buffer": 2,
                            "name": "filterTensor",
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
                            "builtin_options_type": "Conv2DOptions",
                            "builtin_options": {
                                "padding": "VALID",
                                "stride_w": 1,
                                "stride_h": 1,
                                "fused_activation_function": "NONE"
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { },
                    { "data": [ 2,1,0,  6,2,1, 4,1,2 ], },
                    { },
                ]
            }
        )";
        ReadStringToBinary();
    }

    void CheckBufferContents(const TfLiteParser::ModelPtr& model,
                             std::vector<int32_t> bufferValues, size_t bufferIndex)
    {
        for(long unsigned int i=0; i<bufferValues.size(); i++)
        {
            BOOST_CHECK_EQUAL(TfLiteParser::GetBuffer(model, bufferIndex)->data[i], bufferValues[i]);
        }
    }
};

BOOST_FIXTURE_TEST_CASE(GetBufferCheckContents, GetBufferFixture)
{
    //Check contents of buffer are correct
    TfLiteParser::ModelPtr model = TfLiteParser::LoadModelFromBinary(m_GraphBinary.data(), m_GraphBinary.size());
    std::vector<int32_t> bufferValues = {2,1,0,6,2,1,4,1,2};
    CheckBufferContents(model, bufferValues, 2);
}

BOOST_FIXTURE_TEST_CASE(GetBufferCheckEmpty, GetBufferFixture)
{
    //Check if test fixture buffers are empty or not
    TfLiteParser::ModelPtr model = TfLiteParser::LoadModelFromBinary(m_GraphBinary.data(), m_GraphBinary.size());
    BOOST_CHECK(TfLiteParser::GetBuffer(model, 0)->data.empty());
    BOOST_CHECK(TfLiteParser::GetBuffer(model, 1)->data.empty());
    BOOST_CHECK(!TfLiteParser::GetBuffer(model, 2)->data.empty());
    BOOST_CHECK(TfLiteParser::GetBuffer(model, 3)->data.empty());
}

BOOST_FIXTURE_TEST_CASE(GetBufferCheckParseException, GetBufferFixture)
{
    //Check if armnn::ParseException thrown when invalid buffer index used
    TfLiteParser::ModelPtr model = TfLiteParser::LoadModelFromBinary(m_GraphBinary.data(), m_GraphBinary.size());
    BOOST_CHECK_THROW(TfLiteParser::GetBuffer(model, 4), armnn::Exception);
}

BOOST_AUTO_TEST_SUITE_END()
