//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <catch.hpp>
#include <limits>

#include "MathUtils.hpp"
#include <iostream>
#include <numeric>

TEST_CASE("Test DotProductF32")
{
    // Test  Constants:
    const int length = 6;

    float inputA[] = { 1, 1, 1, 0, 0, 0 };
    float inputB[] = { 0, 0, 0, 1, 1, 1 };

    float dot_prod = MathUtils::DotProductF32(inputA, inputB, length);
    float expectedResult = 0;
    CHECK(dot_prod == expectedResult);
}

TEST_CASE("Test FFT32")
{
    // Test  Constants:
    std::vector<float> input(32, 0);
    std::vector<float> output(32);
    std::vector<float> expectedResult(32, 0);

    MathUtils::FftF32(input, output);

    // To avoid common failed assertions due to rounding of near-zero values a small offset is added
    transform(output.begin(), output.end(), output.begin(),
    bind2nd(std::plus<double>(), 0.1));

    transform(expectedResult.begin(), expectedResult.end(), expectedResult.begin(),
    bind2nd(std::plus<double>(), 0.1));

    for (int i = 0; i < output.size(); i++)
    {
        CHECK (expectedResult[i] == Approx(output[i]));
    }
}

TEST_CASE("Test ComplexMagnitudeSquaredF32")
{
    // Test  Constants:
    float input[] = { 0.0, 0.0, 0.5, 0.5,1,1 };
    int inputLen = (sizeof(input)/sizeof(*input));
    float expectedResult[] = { 0.0, 0.5, 2 };
    int outputLen = inputLen/2;
    float output[outputLen];

    MathUtils::ComplexMagnitudeSquaredF32(input, inputLen, output, outputLen);

    for (int i = 0; i < outputLen; i++)
    {
        CHECK (expectedResult[i] == Approx(output[i]));
    }
}

TEST_CASE("Test VecLogarithmF32")
{
    // Test  Constants:

    std::vector<float> input = { 1, 0.1e-10 };
    std::vector<float> expectedResult = { 0, -25.328436 };
    std::vector<float> output(input.size());
    MathUtils::VecLogarithmF32(input,output);

    for (int i = 0; i < input.size(); i++)
    {
        CHECK (expectedResult[i] == Approx(output[i]));
    }
}

TEST_CASE("Test MeanF32")
{    
    // Test  Constants:
    float input[] = { 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000 };
    uint32_t inputLen = (sizeof(input)/sizeof(*input));
    float output;

    // Manually calculated mean of above array
    float expectedResult = 0.100;
    output = MathUtils::MeanF32(input, inputLen);

    CHECK (expectedResult == Approx(output));
}

TEST_CASE("Test StdDevF32")
{
    // Test  Constants:

    float input[] = { 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000 };

    uint32_t inputLen = (sizeof(input)/sizeof(*input));

    // Calculate mean using std library to avoid dependency on MathUtils::MeanF32 
    float mean = (std::accumulate(input, input + inputLen, 0.0f))/float(inputLen);

    float output = MathUtils::StdDevF32(input, inputLen, mean);

    // Manually calculated standard deviation of above array
    float expectedResult = 0.300;

    CHECK (expectedResult == Approx(output));
}

