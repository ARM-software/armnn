//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnnUtils/FloatingPointConverter.hpp>

#include <Half.hpp>

#include <vector>

#include <doctest/doctest.h>

TEST_SUITE("TestFPConversion")
{
TEST_CASE("TestConvertFp32ToFp16")
{
    using namespace half_float::literal;

    float floatArray[] = { 1.0f, 2.0f, 0.5f, 3.1f, 2.4f,
                           5.666f, 6.444f, 7.1f, 432.121f, 12.22f };
    size_t numFloats = sizeof(floatArray) / sizeof(floatArray[0]);
    std::vector<armnn::Half> convertedBuffer(numFloats, 0.0_h);

    armnnUtils::FloatingPointConverter::ConvertFloat32To16(floatArray, numFloats, convertedBuffer.data());

    for (size_t i = 0; i < numFloats; i++)
    {
        armnn::Half expected(floatArray[i]);
        armnn::Half actual = convertedBuffer[i];
        CHECK_EQ(expected, actual);

        float convertedHalf = actual;
        CHECK_EQ(floatArray[i], doctest::Approx(convertedHalf).epsilon(0.07));
    }
}

TEST_CASE("TestConvertFp16ToFp32")
{
    using namespace half_float::literal;

    armnn::Half halfArray[] = { 1.0_h, 2.0_h, 0.5_h, 3.1_h, 2.4_h,
                                5.666_h, 6.444_h, 7.1_h, 432.121_h, 12.22_h };
    size_t numFloats = sizeof(halfArray) / sizeof(halfArray[0]);
    std::vector<float> convertedBuffer(numFloats, 0.0f);

    armnnUtils::FloatingPointConverter::ConvertFloat16To32(halfArray, numFloats, convertedBuffer.data());

    for (size_t i = 0; i < numFloats; i++)
    {
        float expected(halfArray[i]);
        float actual = convertedBuffer[i];
        CHECK_EQ(expected, actual);
    }
}

}
