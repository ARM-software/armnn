//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnnUtils/FloatingPointConverter.hpp>

#include <BFloat16.hpp>
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

TEST_CASE("TestConvertFloat32ToBFloat16")
{
    float floatArray[] = { 1.704735E38f,   // 0x7F004000 round down
                           0.0f,           // 0x00000000 round down
                           2.2959E-41f,    // 0x00004000 round down
                           1.7180272E38f,  // 0x7F014000 round down
                           9.18355E-41f,   // 0x00010000 round down
                           1.14794E-40f,   // 0x00014000 round down
                           4.5918E-41f,    // 0x00008000 round down
                           -1.708058E38f,  // 0xFF008000 round down
                           -4.3033756E37f, // 0xFE018000 round up
                           1.60712E-40f,   // 0x0001C000 round up
                           -2.0234377f,    // 0xC0018001 round up
                           -1.1800863E-38f,// 0x80808001 round up
                           4.843037E-35f,  // 0x0680C000 round up
                           3.9999998f,     // 0x407FFFFF round up
                           std::numeric_limits<float>::max(),    // 0x7F7FFFFF max positive value
                           std::numeric_limits<float>::lowest(), // 0xFF7FFFFF max negative value
                           1.1754942E-38f, // 0x007FFFFF min positive value
                           -1.1754942E-38f // 0x807FFFFF min negative value
                          };
    uint16_t expectedResult[] = { 0x7F00,
                                  0x0000,
                                  0x0000,
                                  0x7F01,
                                  0x0001,
                                  0x0001,
                                  0x0000,
                                  0xFF00,
                                  0xFE02,
                                  0x0002,
                                  0xC002,
                                  0x8081,
                                  0x0681,
                                  0x4080,
                                  0x7F80,
                                  0xFF80,
                                  0x0080,
                                  0x8080
                                 };
    size_t numFloats = sizeof(floatArray) / sizeof(floatArray[0]);

    std::vector<armnn::BFloat16> convertedBuffer(numFloats);

    armnnUtils::FloatingPointConverter::ConvertFloat32ToBFloat16(floatArray, numFloats, convertedBuffer.data());

    for (size_t i = 0; i < numFloats; i++)
    {
        armnn::BFloat16 actual = convertedBuffer[i];
        CHECK_EQ(expectedResult[i], actual.Val());
    }
}

TEST_CASE("TestConvertBFloat16ToFloat32")
{
    uint16_t bf16Array[] = { 16256, 16320, 38699, 16384, 49156, 32639 };
    size_t numFloats = sizeof(bf16Array) / sizeof(bf16Array[0]);
    float expectedResult[] = { 1.0f, 1.5f, -5.525308E-25f, 2.0f, -2.0625f, 3.3895314E38f };
    std::vector<float> convertedBuffer(numFloats, 0.0f);

    armnnUtils::FloatingPointConverter::ConvertBFloat16ToFloat32(bf16Array, numFloats, convertedBuffer.data());

    for (size_t i = 0; i < numFloats; i++)
    {
        float actual = convertedBuffer[i];
        CHECK_EQ(expectedResult[i], actual);
    }
}

}
