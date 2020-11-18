//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
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

void CompareData(std::vector<bool>& tensor1, bool tensor2[], size_t tensorSize)
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

void CompareData(int8_t tensor1[], int8_t tensor2[], size_t tensorSize)
{
    int8_t tolerance = 1;
    for (size_t i = 0; i < tensorSize; i++)
    {
        CHECK(std::max(tensor1[i], tensor2[i]) - std::min(tensor1[i], tensor2[i]) <= tolerance);
    }
}

} // namespace armnnDelegate