//
// Copyright © 2019,2021-2023 Arm Ltd and Contributors. All rights reserved.
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

TEST_CASE("ExpandDimsBy1Rank")
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Expand by 1 dimension
    armnn::TensorShape outputShape = ExpandDimsToRank(inputShape, 4);
    CHECK(outputShape.GetNumDimensions() == 4);
    CHECK(outputShape[0] == 1);
    CHECK(outputShape[1] == 2);
    CHECK(outputShape[2] == 3);
    CHECK(outputShape[3] == 4);
}

TEST_CASE("ExpandDimsBy2Ranks")
{
    armnn::TensorShape inputShape({ 3, 4 });

    // Expand 2 dimensions
    armnn::TensorShape outputShape = ExpandDimsToRank(inputShape, 4);
    CHECK(outputShape.GetNumDimensions() == 4);
    CHECK(outputShape[0] == 1);
    CHECK(outputShape[1] == 1);
    CHECK(outputShape[2] == 3);
    CHECK(outputShape[3] == 4);
}

TEST_CASE("ExpandDimsBy3Ranks")
{
    armnn::TensorShape inputShape({ 4 });

    // Expand 3 dimensions
    armnn::TensorShape outputShape = ExpandDimsToRank(inputShape, 4);
    CHECK(outputShape.GetNumDimensions() == 4);
    CHECK(outputShape[0] == 1);
    CHECK(outputShape[1] == 1);
    CHECK(outputShape[2] == 1);
    CHECK(outputShape[3] == 4);
}

TEST_CASE("ExpandDimsInvalidRankAmount")
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Don't expand because target rank is smaller than current rank
    armnn::TensorShape outputShape = ExpandDimsToRank(inputShape, 2);
    CHECK(outputShape.GetNumDimensions() == 3);
    CHECK(outputShape[0] == 2);
    CHECK(outputShape[1] == 3);
    CHECK(outputShape[2] == 4);
}

TEST_CASE("ExpandDimsToRankInvalidTensorShape")
{
    armnn::TensorShape inputShape({ 2, 3, 4 });

    // Throw exception because rank 6 tensors are unsupported by armnn
    CHECK_THROWS_AS(ExpandDimsToRank(inputShape, 6), armnn::InvalidArgumentException);
}


TEST_CASE("ReduceDimsShapeAll1s")
{
    armnn::TensorShape inputShape({ 1, 1, 1 });

    // Reduce dimension 2
    armnn::TensorShape outputShape = ReduceDims(inputShape, 2);
    CHECK(outputShape.GetNumDimensions() == 2);
    CHECK(outputShape[0] == 1);
    CHECK(outputShape[1] == 1);
}

TEST_CASE("ReduceDimsShapeNotEnough1s")
{
    armnn::TensorShape inputShape({ 1, 2, 1 });

    // Reduce dimension 1
    armnn::TensorShape outputShape = ReduceDims(inputShape, 1);
    CHECK(outputShape.GetNumDimensions() == 2);
    CHECK(outputShape[0] == 2);
    CHECK(outputShape[1] == 1);
}

TEST_CASE("ReduceDimsInfoAll1s")
{
    armnn::TensorInfo inputInfo({ 1, 1, 1 }, DataType::Float32);

    // Reduce dimension 2
    armnn::TensorInfo outputInfo = ReduceDims(inputInfo, 2);
    CHECK(outputInfo.GetShape().GetNumDimensions() == 2);
    CHECK(outputInfo.GetShape()[0] == 1);
    CHECK(outputInfo.GetShape()[1] == 1);
}

TEST_CASE("ReduceDimsInfoNotEnough1s")
{
    armnn::TensorInfo inputInfo({ 1, 2, 1 }, DataType::Float32);

    // Reduce dimension 1
    armnn::TensorInfo outputInfo = ReduceDims(inputInfo, 1);
    CHECK(outputInfo.GetNumDimensions() == 2);
    CHECK(outputInfo.GetShape()[0] == 2);
    CHECK(outputInfo.GetShape()[1] == 1);
}

TEST_CASE("ReduceDimsShapeDimensionGreaterThanSize")
{
    armnn::TensorShape inputShape({ 1, 1, 1 });

    // Do not reduce because dimension does not exist
    armnn::TensorShape outputShape = ReduceDims(inputShape, 4);
    CHECK(outputShape.GetNumDimensions() == 3);
    CHECK(outputShape[0] == 1);
    CHECK(outputShape[1] == 1);
    CHECK(outputShape[2] == 1);
}


TEST_CASE("ToFloatArrayInvalidDataType")
{
    armnn::TensorInfo info({ 2, 3, 4 }, armnn::DataType::BFloat16);
    std::vector<uint8_t> data {1,2,3,4,5,6,7,8,9,10};

    // Invalid argument
    CHECK_THROWS_AS(ToFloatArray(data, info), armnn::InvalidArgumentException);
}

TEST_CASE("ToFloatArrayQSymmS8PerAxis")
{
    std::vector<float> quantizationScales { 0.1f, 0.2f, 0.3f, 0.4f };
    unsigned int quantizationDim = 1;

    armnn::TensorInfo info({ 3, 4 }, armnn::DataType::QSymmS8, quantizationScales, quantizationDim);
    std::vector<uint8_t> data { 100, 120, 130, 140, 150, 160, 170 ,180, 190, 200, 210, 220 };
    float expected[] { 10.0f, 24.0f, -37.8f, -46.4f, -10.6f, -19.2f, -25.8f, -30.4f, -6.6f, -11.2f, -13.8f, -14.4f };

    std::unique_ptr<float[]> result = ToFloatArray(data, info);

    for (uint i = 0; i < info.GetNumElements(); ++i)
    {
        CHECK_EQ(result[i], doctest::Approx(expected[i]));
    }
}

TEST_CASE("ToFloatArrayQSymmS8")
{
    armnn::TensorInfo info({ 3, 4 }, armnn::DataType::QSymmS8, 0.1f);
    std::vector<uint8_t> data { 100, 120, 130, 140, 150, 160, 170 ,180, 190, 200, 210, 220 };
    float expected[] { 10.0f, 12.0f, -12.6f, -11.6f, -10.6f, -9.6f, -8.6f, -7.6f, -6.6f,  -5.6f, -4.6f, -3.6f };

    std::unique_ptr<float[]> result = ToFloatArray(data, info);

    for (uint i = 0; i < info.GetNumElements(); ++i)
    {
        CHECK_EQ(result[i], doctest::Approx(expected[i]));
    }
}

TEST_CASE("ToFloatArrayQAsymmS8PerAxis")
{
    std::vector<float> quantizationScales { 0.1f, 0.2f, 0.3f, 0.4f };
    unsigned int quantizationDim = 1;

    armnn::TensorInfo info({ 3, 4 }, armnn::DataType::QAsymmS8, quantizationScales, quantizationDim);
    std::vector<uint8_t> data { 100, 120, 130, 140, 150, 160, 170 ,180, 190, 200, 210, 220 };
    float expected[] { 10.0f, 24.0f, -37.8f, -46.4f, -10.6f, -19.2f, -25.8f, -30.4f, -6.6f, -11.2f, -13.8f, -14.4f };

    std::unique_ptr<float[]> result = ToFloatArray(data, info);

    for (uint i = 0; i < info.GetNumElements(); ++i)
    {
        CHECK_EQ(result[i], doctest::Approx(expected[i]));
    }
}

TEST_CASE("ToFloatArrayQAsymmS8")
{
    armnn::TensorInfo info({ 3, 4 }, armnn::DataType::QAsymmS8, 0.1f);
    std::vector<uint8_t> data { 100, 120, 130, 140, 150, 160, 170 ,180, 190, 200, 210, 220 };
    float expected[] { 10.0f, 12.0f, -12.6f, -11.6f, -10.6f, -9.6f, -8.6f, -7.6f, -6.6f,  -5.6f, -4.6f, -3.6f };

    std::unique_ptr<float[]> result = ToFloatArray(data, info);

    for (uint i = 0; i < info.GetNumElements(); ++i)
    {
        CHECK_EQ(result[i], doctest::Approx(expected[i]));
    }
}

TEST_CASE("ToFloatArrayQASymmU8PerAxis")
{
    std::vector<float> quantizationScales { 0.1f, 0.2f, 0.3f, 0.4f };
    unsigned int quantizationDim = 1;

    armnn::TensorInfo info({ 3, 4 }, armnn::DataType::QAsymmU8, quantizationScales, quantizationDim);
    std::vector<uint8_t> data { 100, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220 };
    float expected[] { 10.0f, 24.0f, 39.0f, 56.0f, 15.0f, 32.0f, 51.0f, 72.0f, 19.0f, 40.0f, 63.0f, 88.0f };

    std::unique_ptr<float[]> result = ToFloatArray(data, info);

    for (uint i = 0; i < info.GetNumElements(); ++i)
    {
        CHECK_EQ(result[i], doctest::Approx(expected[i]));
    }
}

TEST_CASE("ToFloatArrayQAsymmU8")
{
    armnn::TensorInfo info({ 3, 4 }, armnn::DataType::QAsymmU8, 0.1f);
    std::vector<uint8_t> data { 100, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220 };
    float expected[] { 10.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f };

    std::unique_ptr<float[]> result = ToFloatArray(data, info);

    for (uint i = 0; i < info.GetNumElements(); ++i)
    {
        CHECK_EQ(result[i], doctest::Approx(expected[i]));
    }
}

TEST_CASE("ToFloatArraySigned32PerAxis")
{
    std::vector<float> quantizationScales { 0.1f, 0.2f, 0.3f, 0.4f };
    unsigned int quantizationDim = 1;

    armnn::TensorInfo info({ 3, 4 }, armnn::DataType::Signed32, quantizationScales, quantizationDim);
    std::vector<uint8_t> data { 100, 0, 0, 0, 120, 0, 0, 0, 130, 0, 0, 0, 140, 0, 0, 0, 150, 0, 0, 0, 160, 0, 0, 0,
                                170, 0, 0, 0, 180, 0, 0, 0, 190, 0, 0, 0, 200, 0, 0, 0, 210, 0, 0, 0, 220, 0, 0, 0 };
    float expected[] { 10.0f, 24.0f, 39.0f, 56.0f, 15.0f, 32.0f, 51.0f, 72.0f, 19.0f, 40.0f, 63.0f, 88.0f };

    std::unique_ptr<float[]> result = ToFloatArray(data, info);

    for (uint i = 0; i < info.GetNumElements(); ++i)
    {
        CHECK_EQ(result[i], doctest::Approx(expected[i]));
    }
}

TEST_CASE("ToFloatArraySigned32")
{
    armnn::TensorInfo info({ 3, 4 }, armnn::DataType::Signed32, 0.1f);
    std::vector<uint8_t> data { 100, 0, 0, 0, 120, 0, 0, 0, 130, 0, 0, 0, 140, 0, 0, 0, 150, 0, 0, 0, 160, 0, 0, 0,
                                170, 0, 0, 0, 180, 0, 0, 0, 190, 0, 0, 0, 200, 0, 0, 0, 210, 0, 0, 0, 220, 0, 0, 0 };
    float expected[] { 10.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f };

    std::unique_ptr<float[]> result = ToFloatArray(data, info);

    for (uint i = 0; i < info.GetNumElements(); ++i)
    {
        CHECK_EQ(result[i], doctest::Approx(expected[i]));
    }
}

TEST_CASE("ToFloatArraySigned64PerAxis")
{
    std::vector<float> quantizationScales { 0.1f, 0.2f, 0.3f, 0.4f };
    unsigned int quantizationDim = 1;

    armnn::TensorInfo info({ 3, 4 }, armnn::DataType::Signed64, quantizationScales, quantizationDim);
    std::vector<uint8_t> data { 100, 0, 0, 0, 0, 0, 0, 0, 120, 0, 0, 0, 0, 0, 0, 0, 130, 0, 0, 0, 0, 0, 0, 0,
                                140, 0, 0, 0, 0, 0, 0, 0, 150, 0, 0, 0, 0, 0, 0, 0, 160, 0, 0, 0, 0, 0, 0, 0,
                                170, 0, 0, 0, 0, 0, 0, 0, 180, 0, 0, 0, 0, 0, 0, 0, 190, 0, 0, 0, 0, 0, 0, 0,
                                200, 0, 0, 0, 0, 0, 0, 0, 210, 0, 0, 0, 0, 0, 0, 0, 220, 0, 0, 0, 0, 0, 0, 0 };
    float expected[] { 10.0f, 24.0f, 39.0f, 56.0f, 15.0f, 32.0f, 51.0f, 72.0f, 19.0f, 40.0f, 63.0f, 88.0f };

    std::unique_ptr<float[]> result = ToFloatArray(data, info);

    for (uint i = 0; i < info.GetNumElements(); ++i)
    {
        CHECK_EQ(result[i], doctest::Approx(expected[i]));
    }
}

TEST_CASE("ToFloatArraySigned64")
{
    armnn::TensorInfo info({ 3, 4 }, armnn::DataType::Signed64, 0.1f);
    std::vector<uint8_t> data { 100, 0, 0, 0, 0, 0, 0, 0, 120, 0, 0, 0, 0, 0, 0, 0, 130, 0, 0, 0, 0, 0, 0, 0,
                                140, 0, 0, 0, 0, 0, 0, 0, 150, 0, 0, 0, 0, 0, 0, 0, 160, 0, 0, 0, 0, 0, 0, 0,
                                170, 0, 0, 0, 0, 0, 0, 0, 180, 0, 0, 0, 0, 0, 0, 0, 190, 0, 0, 0, 0, 0, 0, 0,
                                200, 0, 0, 0, 0, 0, 0, 0, 210, 0, 0, 0, 0, 0, 0, 0, 220, 0, 0, 0, 0, 0, 0, 0 };
    float expected[] { 10.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f };

    std::unique_ptr<float[]> result = ToFloatArray(data, info);

    for (uint i = 0; i < info.GetNumElements(); ++i)
    {
        CHECK_EQ(result[i], doctest::Approx(expected[i]));
    }
}
}
