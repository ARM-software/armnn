//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "../TfLiteParser.hpp"
#include <iostream>
#include <string>

struct TfLiteParserFixture
{

    armnnTfLiteParser::TfLiteParser m_Parser;
    unsigned int m_InputShape[4];

    TfLiteParserFixture() : m_Parser( ), m_InputShape { 1, 2, 2, 1 } {
        m_Parser.Create();
    }
    ~TfLiteParserFixture()          {  }

};

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser);


BOOST_FIXTURE_TEST_CASE( EmptySqueezeDims_OutputWithAllDimensionsSqueezed, TfLiteParserFixture )
{

    std::vector<uint32_t> squeezeDims = {  };

    armnn::TensorInfo inputTensorInfo = armnn::TensorInfo(4, m_InputShape, armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo = m_Parser.OutputShapeOfSqueeze(squeezeDims, inputTensorInfo);
    BOOST_TEST(outputTensorInfo.GetNumElements() == 4);
    BOOST_TEST(outputTensorInfo.GetNumDimensions() == 2);
    BOOST_TEST((outputTensorInfo.GetShape() == armnn::TensorShape({ 2, 2 })));
};

BOOST_FIXTURE_TEST_CASE( SqueezeDimsNotIncludingSizeOneDimensions_NoDimensionsSqueezedInOutput, TfLiteParserFixture )
{
    std::vector<uint32_t> squeezeDims = { 1, 2 };

    armnn::TensorInfo inputTensorInfo = armnn::TensorInfo(4, m_InputShape, armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo = m_Parser.OutputShapeOfSqueeze(squeezeDims, inputTensorInfo);
    BOOST_TEST(outputTensorInfo.GetNumElements() == 4);
    BOOST_TEST(outputTensorInfo.GetNumDimensions() == 4);
    BOOST_TEST((outputTensorInfo.GetShape() == armnn::TensorShape({ 1, 2, 2, 1 })));
};

BOOST_FIXTURE_TEST_CASE( SqueezeDimsRangePartial_OutputWithDimensionsWithinRangeSqueezed, TfLiteParserFixture )
{
    std::vector<uint32_t> squeezeDims = { 1, 3 };

    armnn::TensorInfo inputTensorInfo = armnn::TensorInfo(4, m_InputShape, armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo = m_Parser.OutputShapeOfSqueeze(squeezeDims, inputTensorInfo);
    BOOST_TEST(outputTensorInfo.GetNumElements() == 4);
    BOOST_TEST(outputTensorInfo.GetNumDimensions() == 3);
    BOOST_TEST((outputTensorInfo.GetShape() == armnn::TensorShape({ 1, 2, 2 })));
};

BOOST_AUTO_TEST_SUITE_END();