//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <reference/workloads/Decoders.hpp>

#include <fmt/format.h>

#include <doctest/doctest.h>

TEST_SUITE("RefPerChannelDecoder")
{
template<typename T>
void CompareVector(std::vector<T> vec1, std::vector<T> vec2)
{
    CHECK(vec1.size() == vec2.size());

    bool mismatch = false;
    for (uint32_t i = 0; i < vec1.size(); ++i)
    {
        if (vec1[i] != vec2[i])
        {
            MESSAGE(fmt::format("Vector value mismatch: index={}  {} != {}",
                                i,
                                vec1[i],
                                vec2[i]));

            mismatch = true;
        }
    }

    if (mismatch)
    {
        FAIL("Error in CompareVector. Vectors don't match.");
    }
}

// Ensure quantization works for none depthwise convolutions
TEST_CASE("RefPerChannelDecoderTest1")
{
    using namespace armnn;
    std::vector<int8_t> input =
    {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    };

    std::vector<float> expOutput =
    {
        0.0f,   1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f, 10.0f, 11.0f,
        24.0f, 26.0f, 28.0f, 30.0f, 32.0f, 34.0f, 36.0f, 38.0f, 40.0f, 42.0f, 44.0f, 46.0f
    };

    TensorInfo tensorInfo ({2,2,2,3},DataType::QSymmS8,{1.0f, 2.0f},0);
    auto decoder = MakeDecoder<float>(tensorInfo, input.data());

    std::vector<float> output = decoder->DecodeTensor(tensorInfo.GetShape());

    CompareVector(output, expOutput);
}

// Ensure quantization works for depthwise convolutions M=1
TEST_CASE("RefPerChannelDecoderTest2")
{
    using namespace armnn;
    std::vector<int8_t> input =
    {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    };

    std::vector<float> expOutput =
    {
         0.0f,  1.0f,  2.0f,  3.0f,
         8.0f, 10.0f, 12.0f, 14.0f,
        24.0f, 27.0f, 30.0f, 33.0f,
        48.0f, 52.0f, 56.0f, 60.0f
    };

    // [O,1,H,W] = [I*M,1,H,W] = [4*1,1,2,2]
    TensorInfo tensorInfo ({4,1,2,2},DataType::QSymmS8,{1.0f, 2.0f, 3.0f, 4.0f},0);
    auto decoder = MakeDecoder<float>(tensorInfo, input.data());

    std::vector<float> output = decoder->DecodeTensor(tensorInfo.GetShape(), true);

    CompareVector(output, expOutput);
}

// Ensure quantization works for depthwise convolutions M=2
TEST_CASE("RefPerChannelDecoderTest3")
{
    using namespace armnn;
    std::vector<int8_t> input =
    {
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23
    };

    std::vector<float> expOutput =
    {
         0.0f,  1.0f,  2.0f,  3.0f,
         8.0f, 10.0f, 12.0f, 14.0f,
        24.0f, 27.0f, 30.0f, 33.0f,
        48.0f, 52.0f, 56.0f, 60.0f,
        80.0f, 85.0f, 90.0f, 95.0f,
        120.0f, 126.0f, 132.0f, 138.0f
    };

    // [O,1,H,W] = [I*M,1,H,W] = [3*2,1,2,2]
    TensorInfo tensorInfo ({6,1,2,2},DataType::QSymmS8,{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},0);
    auto decoder = MakeDecoder<float>(tensorInfo, input.data());

    std::vector<float> output = decoder->DecodeTensor(tensorInfo.GetShape(), true);

    CompareVector(output, expOutput);
}

// Ensure quantization works for depthwise convolutions M=2 for int32
TEST_CASE("RefPerChannelDecoderTest4")
{
    using namespace armnn;
    std::vector<int32_t> input =
    {
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23
    };

    std::vector<float> expOutput =
    {
         0.0f,  1.0f,  2.0f,  3.0f,
         8.0f, 10.0f, 12.0f, 14.0f,
        24.0f, 27.0f, 30.0f, 33.0f,
        48.0f, 52.0f, 56.0f, 60.0f,
        80.0f, 85.0f, 90.0f, 95.0f,
        120.0f, 126.0f, 132.0f, 138.0f
    };

    // [O,1,H,W] = [I*M,1,H,W] = [3*2,1,2,2]
    TensorInfo tensorInfo ({6,1,2,2},DataType::Signed32,{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},0);
    auto decoder = MakeDecoder<float>(tensorInfo, input.data());

    std::vector<float> output = decoder->DecodeTensor(tensorInfo.GetShape(), true);

    CompareVector(output, expOutput);
}

}
