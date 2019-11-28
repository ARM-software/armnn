//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Types.hpp>

#include <armnnUtils/TensorUtils.hpp>

#include <boost/test/unit_test.hpp>

using namespace armnn;
using namespace armnnUtils;

BOOST_AUTO_TEST_SUITE(TensorUtilsSuite)

BOOST_AUTO_TEST_CASE(ExpandDimsAxis0Test)
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Expand dimension 0
    armnn::TensorShape outputShape = ExpandDims(inputShape, 0);
    BOOST_TEST(outputShape.GetNumDimensions() == 4);
    BOOST_TEST(outputShape[0] == 1);
    BOOST_TEST(outputShape[1] == 2);
    BOOST_TEST(outputShape[2] == 3);
    BOOST_TEST(outputShape[3] == 4);
}

BOOST_AUTO_TEST_CASE(ExpandDimsAxis1Test)
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Expand dimension 1
    armnn::TensorShape outputShape = ExpandDims(inputShape, 1);
    BOOST_TEST(outputShape.GetNumDimensions() == 4);
    BOOST_TEST(outputShape[0] == 2);
    BOOST_TEST(outputShape[1] == 1);
    BOOST_TEST(outputShape[2] == 3);
    BOOST_TEST(outputShape[3] == 4);
}

BOOST_AUTO_TEST_CASE(ExpandDimsAxis2Test)
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Expand dimension 2
    armnn::TensorShape outputShape = ExpandDims(inputShape, 2);
    BOOST_TEST(outputShape.GetNumDimensions() == 4);
    BOOST_TEST(outputShape[0] == 2);
    BOOST_TEST(outputShape[1] == 3);
    BOOST_TEST(outputShape[2] == 1);
    BOOST_TEST(outputShape[3] == 4);
}

BOOST_AUTO_TEST_CASE(ExpandDimsAxis3Test)
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Expand dimension 3
    armnn::TensorShape outputShape = ExpandDims(inputShape, 3);
    BOOST_TEST(outputShape.GetNumDimensions() == 4);
    BOOST_TEST(outputShape[0] == 2);
    BOOST_TEST(outputShape[1] == 3);
    BOOST_TEST(outputShape[2] == 4);
    BOOST_TEST(outputShape[3] == 1);
}

BOOST_AUTO_TEST_CASE(ExpandDimsNegativeAxis1Test)
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Expand dimension -1
    armnn::TensorShape outputShape = ExpandDims(inputShape, -1);
    BOOST_TEST(outputShape.GetNumDimensions() == 4);
    BOOST_TEST(outputShape[0] == 2);
    BOOST_TEST(outputShape[1] == 3);
    BOOST_TEST(outputShape[2] == 4);
    BOOST_TEST(outputShape[3] == 1);
}

BOOST_AUTO_TEST_CASE(ExpandDimsNegativeAxis2Test)
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Expand dimension -2
    armnn::TensorShape outputShape = ExpandDims(inputShape, -2);
    BOOST_TEST(outputShape.GetNumDimensions() == 4);
    BOOST_TEST(outputShape[0] == 2);
    BOOST_TEST(outputShape[1] == 3);
    BOOST_TEST(outputShape[2] == 1);
    BOOST_TEST(outputShape[3] == 4);
}

BOOST_AUTO_TEST_CASE(ExpandDimsNegativeAxis3Test)
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Expand dimension -3
    armnn::TensorShape outputShape = ExpandDims(inputShape, -3);
    BOOST_TEST(outputShape.GetNumDimensions() == 4);
    BOOST_TEST(outputShape[0] == 2);
    BOOST_TEST(outputShape[1] == 1);
    BOOST_TEST(outputShape[2] == 3);
    BOOST_TEST(outputShape[3] == 4);
}

BOOST_AUTO_TEST_CASE(ExpandDimsNegativeAxis4Test)
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Expand dimension -4
    armnn::TensorShape outputShape = ExpandDims(inputShape, -4);
    BOOST_TEST(outputShape.GetNumDimensions() == 4);
    BOOST_TEST(outputShape[0] == 1);
    BOOST_TEST(outputShape[1] == 2);
    BOOST_TEST(outputShape[2] == 3);
    BOOST_TEST(outputShape[3] == 4);
}

BOOST_AUTO_TEST_CASE(ExpandDimsInvalidAxisTest)
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Invalid expand dimension 4
    BOOST_CHECK_THROW(ExpandDims(inputShape, 4), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(ExpandDimsInvalidNegativeAxisTest)
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Invalid expand dimension -5
    BOOST_CHECK_THROW(ExpandDims(inputShape, -5), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_SUITE_END()
