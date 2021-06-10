//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Types.hpp>

#include <armnnUtils/TensorUtils.hpp>

#include <doctest/doctest.h>

using namespace armnn;
using namespace armnnUtils;

TEST_SUITE("TensorUtilsSuite")
{
TEST_CASE("ExpandDimsAxis0Test")
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Expand dimension 0
    armnn::TensorShape outputShape = ExpandDims(inputShape, 0);
    CHECK(outputShape.GetNumDimensions() == 4);
    CHECK(outputShape[0] == 1);
    CHECK(outputShape[1] == 2);
    CHECK(outputShape[2] == 3);
    CHECK(outputShape[3] == 4);
}

TEST_CASE("ExpandDimsAxis1Test")
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Expand dimension 1
    armnn::TensorShape outputShape = ExpandDims(inputShape, 1);
    CHECK(outputShape.GetNumDimensions() == 4);
    CHECK(outputShape[0] == 2);
    CHECK(outputShape[1] == 1);
    CHECK(outputShape[2] == 3);
    CHECK(outputShape[3] == 4);
}

TEST_CASE("ExpandDimsAxis2Test")
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Expand dimension 2
    armnn::TensorShape outputShape = ExpandDims(inputShape, 2);
    CHECK(outputShape.GetNumDimensions() == 4);
    CHECK(outputShape[0] == 2);
    CHECK(outputShape[1] == 3);
    CHECK(outputShape[2] == 1);
    CHECK(outputShape[3] == 4);
}

TEST_CASE("ExpandDimsAxis3Test")
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Expand dimension 3
    armnn::TensorShape outputShape = ExpandDims(inputShape, 3);
    CHECK(outputShape.GetNumDimensions() == 4);
    CHECK(outputShape[0] == 2);
    CHECK(outputShape[1] == 3);
    CHECK(outputShape[2] == 4);
    CHECK(outputShape[3] == 1);
}

TEST_CASE("ExpandDimsNegativeAxis1Test")
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Expand dimension -1
    armnn::TensorShape outputShape = ExpandDims(inputShape, -1);
    CHECK(outputShape.GetNumDimensions() == 4);
    CHECK(outputShape[0] == 2);
    CHECK(outputShape[1] == 3);
    CHECK(outputShape[2] == 4);
    CHECK(outputShape[3] == 1);
}

TEST_CASE("ExpandDimsNegativeAxis2Test")
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Expand dimension -2
    armnn::TensorShape outputShape = ExpandDims(inputShape, -2);
    CHECK(outputShape.GetNumDimensions() == 4);
    CHECK(outputShape[0] == 2);
    CHECK(outputShape[1] == 3);
    CHECK(outputShape[2] == 1);
    CHECK(outputShape[3] == 4);
}

TEST_CASE("ExpandDimsNegativeAxis3Test")
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Expand dimension -3
    armnn::TensorShape outputShape = ExpandDims(inputShape, -3);
    CHECK(outputShape.GetNumDimensions() == 4);
    CHECK(outputShape[0] == 2);
    CHECK(outputShape[1] == 1);
    CHECK(outputShape[2] == 3);
    CHECK(outputShape[3] == 4);
}

TEST_CASE("ExpandDimsNegativeAxis4Test")
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Expand dimension -4
    armnn::TensorShape outputShape = ExpandDims(inputShape, -4);
    CHECK(outputShape.GetNumDimensions() == 4);
    CHECK(outputShape[0] == 1);
    CHECK(outputShape[1] == 2);
    CHECK(outputShape[2] == 3);
    CHECK(outputShape[3] == 4);
}

TEST_CASE("ExpandDimsInvalidAxisTest")
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Invalid expand dimension 4
    CHECK_THROWS_AS(ExpandDims(inputShape, 4), armnn::InvalidArgumentException);
}

TEST_CASE("ExpandDimsInvalidNegativeAxisTest")
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Invalid expand dimension -5
    CHECK_THROWS_AS(ExpandDims(inputShape, -5), armnn::InvalidArgumentException);
}

}
