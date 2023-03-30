//
// Copyright © 2020, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TestUtils.hpp"

namespace armnnDelegate
{

void CompareData(bool tensor1[], bool tensor2[], size_t tensorSize)
{
    auto compareBool = [](auto a, auto b) {return (((a == 0) && (b == 0)) || ((a != 0) && (b != 0)));};
    for (size_t i = 0; i < tensorSize; i++)
    {
        CHECK(compareBool(tensor1[i], tensor2[i]));
    }
}

void CompareData(std::vector<bool>& tensor1, std::vector<bool>& tensor2, size_t tensorSize)
{
    auto compareBool = [](auto a, auto b) {return (((a == 0) && (b == 0)) || ((a != 0) && (b != 0)));};
    for (size_t i = 0; i < tensorSize; i++)
    {
        CHECK(compareBool(tensor1[i], tensor2[i]));
    }
}

void CompareData(float tensor1[], float tensor2[], size_t tensorSize)
{
    for (size_t i = 0; i < tensorSize; i++)
    {
        CHECK(tensor1[i] == doctest::Approx( tensor2[i] ));
    }
}

void CompareData(float tensor1[], float tensor2[], size_t tensorSize, float percentTolerance)
{
    for (size_t i = 0; i < tensorSize; i++)
    {
        CHECK(std::max(tensor1[i], tensor2[i]) - std::min(tensor1[i], tensor2[i]) <=
              std::abs(tensor1[i]*percentTolerance/100));
    }
}

void CompareData(uint8_t tensor1[], uint8_t tensor2[], size_t tensorSize)
{
    uint8_t tolerance = 1;
    for (size_t i = 0; i < tensorSize; i++)
    {
        CHECK(std::max(tensor1[i], tensor2[i]) - std::min(tensor1[i], tensor2[i]) <= tolerance);
    }
}

void CompareData(int16_t tensor1[], int16_t tensor2[], size_t tensorSize)
{
    int16_t tolerance = 1;
    for (size_t i = 0; i < tensorSize; i++)
    {
        CHECK(std::max(tensor1[i], tensor2[i]) - std::min(tensor1[i], tensor2[i]) <= tolerance);
    }
}

void CompareData(int32_t tensor1[], int32_t tensor2[], size_t tensorSize)
{
    int32_t tolerance = 1;
    for (size_t i = 0; i < tensorSize; i++)
    {
        CHECK(std::max(tensor1[i], tensor2[i]) - std::min(tensor1[i], tensor2[i]) <= tolerance);
    }
}

void CompareData(int8_t tensor1[], int8_t tensor2[], size_t tensorSize)
{
    int8_t tolerance = 1;
    for (size_t i = 0; i < tensorSize; i++)
    {
        CHECK(std::max(tensor1[i], tensor2[i]) - std::min(tensor1[i], tensor2[i]) <= tolerance);
    }
}

void CompareData(Half tensor1[], Half tensor2[], size_t tensorSize)
{
    for (size_t i = 0; i < tensorSize; i++)
    {
        CHECK(tensor1[i] == doctest::Approx( tensor2[i] ));
    }
}

void CompareData(TfLiteFloat16 tensor1[], TfLiteFloat16 tensor2[], size_t tensorSize)
{
    uint16_t tolerance = 1;
    for (size_t i = 0; i < tensorSize; i++)
    {
        uint16_t tensor1Data = tensor1[i].data;
        uint16_t tensor2Data = tensor2[i].data;
        CHECK(std::max(tensor1Data, tensor2Data) - std::min(tensor1Data, tensor2Data) <= tolerance);
    }
}

void CompareData(TfLiteFloat16 tensor1[], Half tensor2[], size_t tensorSize) {
    uint16_t tolerance = 1;
    for (size_t i = 0; i < tensorSize; i++)
    {
        uint16_t tensor1Data = tensor1[i].data;
        uint16_t tensor2Data = half_float::detail::float2half<std::round_indeterminate, float>(tensor2[i]);
        CHECK(std::max(tensor1Data, tensor2Data) - std::min(tensor1Data, tensor2Data) <= tolerance);
    }
}

void CompareOutputShape(const std::vector<int32_t>& tfLiteDelegateShape,
                        const std::vector<int32_t>& armnnDelegateShape,
                        const std::vector<int32_t>& expectedOutputShape)
{
    CHECK(expectedOutputShape.size() == tfLiteDelegateShape.size());
    CHECK(expectedOutputShape.size() == armnnDelegateShape.size());

    for (size_t i = 0; i < expectedOutputShape.size(); i++)
    {
        CHECK(expectedOutputShape[i] == armnnDelegateShape[i]);
        CHECK(tfLiteDelegateShape[i] == expectedOutputShape[i]);
        CHECK(tfLiteDelegateShape[i] == armnnDelegateShape[i]);
    }
}

} // namespace armnnDelegate