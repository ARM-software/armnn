//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../TfLiteParser.hpp"

#include <doctest/doctest.h>

TEST_SUITE("TensorflowLiteParser_OutputShapeOfSqueeze")
{

struct TfLiteParserFixture
{

    armnnTfLiteParser::TfLiteParserImpl m_Parser;
    unsigned int m_InputShape[4];

    TfLiteParserFixture() : m_Parser( ), m_InputShape { 1, 2, 2, 1 } {}
    ~TfLiteParserFixture()          {  }

};

TEST_CASE_FIXTURE(TfLiteParserFixture, "EmptySqueezeDims_OutputWithAllDimensionsSqueezed")
{

    std::vector<uint32_t> squeezeDims = {  };

    armnn::TensorInfo inputTensorInfo = armnn::TensorInfo(4, m_InputShape, armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo = m_Parser.OutputShapeOfSqueeze(squeezeDims, inputTensorInfo);
    CHECK(outputTensorInfo.GetNumElements() == 4);
    CHECK(outputTensorInfo.GetNumDimensions() == 2);
    CHECK((outputTensorInfo.GetShape() == armnn::TensorShape({ 2, 2 })));
};

TEST_CASE_FIXTURE(TfLiteParserFixture, "SqueezeDimsNotIncludingSizeOneDimensions_NoDimensionsSqueezedInOutput")
{
    std::vector<uint32_t> squeezeDims = { 1, 2 };

    armnn::TensorInfo inputTensorInfo = armnn::TensorInfo(4, m_InputShape, armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo = m_Parser.OutputShapeOfSqueeze(squeezeDims, inputTensorInfo);
    CHECK(outputTensorInfo.GetNumElements() == 4);
    CHECK(outputTensorInfo.GetNumDimensions() == 4);
    CHECK((outputTensorInfo.GetShape() == armnn::TensorShape({ 1, 2, 2, 1 })));
};

TEST_CASE_FIXTURE(TfLiteParserFixture, "SqueezeDimsRangePartial_OutputWithDimensionsWithinRangeSqueezed")
{
    std::vector<uint32_t> squeezeDims = { 1, 3 };

    armnn::TensorInfo inputTensorInfo = armnn::TensorInfo(4, m_InputShape, armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo = m_Parser.OutputShapeOfSqueeze(squeezeDims, inputTensorInfo);
    CHECK(outputTensorInfo.GetNumElements() == 4);
    CHECK(outputTensorInfo.GetNumDimensions() == 3);
    CHECK((outputTensorInfo.GetShape() == armnn::TensorShape({ 1, 2, 2 })));
};

}