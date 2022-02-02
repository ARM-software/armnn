//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnnTestUtils/PredicateResult.hpp>

#include <armnn/Tensor.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnnUtils/FloatingPointComparison.hpp>

#include <armnnUtils/QuantizeHelper.hpp>

#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <random>
#include <vector>

constexpr float g_FloatCloseToZeroTolerance = 1.0e-6f;

template<typename T, bool isQuantized = true>
struct SelectiveComparer
{
    static bool Compare(T a, T b)
    {
        return (std::max(a, b) - std::min(a, b)) <= 1;
    }

};

template<typename T>
struct SelectiveComparer<T, false>
{
    static bool Compare(T a, T b)
    {
        // If a or b is zero, percent_tolerance does an exact match, so compare to a small, constant tolerance instead.
        if (a == 0.0f || b == 0.0f)
        {
            return std::abs(a - b) <= g_FloatCloseToZeroTolerance;
        }

        if (std::isinf(a) && a == b)
        {
            return true;
        }

        if (std::isnan(a) && std::isnan(b))
        {
            return true;
        }

        // For unquantized floats we use a tolerance of 1%.
        return armnnUtils::within_percentage_tolerance(a, b);
    }
};

template<typename T>
bool SelectiveCompare(T a, T b)
{
    return SelectiveComparer<T, armnn::IsQuantizedType<T>()>::Compare(a, b);
};

template<typename T>
bool SelectiveCompareBoolean(T a, T b)
{
    return (((a == 0) && (b == 0)) || ((a != 0) && (b != 0)));
};

template <typename T>
armnn::PredicateResult CompareTensors(const std::vector<T>& actualData,
                                      const std::vector<T>& expectedData,
                                      const armnn::TensorShape& actualShape,
                                      const armnn::TensorShape& expectedShape,
                                      bool compareBoolean = false,
                                      bool isDynamic = false)
{
    if (actualData.size() != expectedData.size())
    {
        armnn::PredicateResult res(false);
        res.Message() << "Different data size ["
                      << actualData.size()
                      << "!="
                      << expectedData.size()
                      << "]";
        return res;
    }

    if (actualShape.GetNumDimensions() != expectedShape.GetNumDimensions())
    {
        armnn::PredicateResult res(false);
        res.Message() << "Different number of dimensions ["
                      << actualShape.GetNumDimensions()
                      << "!="
                      << expectedShape.GetNumDimensions()
                      << "]";
        return res;
    }

    if (actualShape.GetNumElements() != expectedShape.GetNumElements())
    {
        armnn::PredicateResult res(false);
        res.Message() << "Different number of elements ["
                      << actualShape.GetNumElements()
                      << "!="
                      << expectedShape.GetNumElements()
                      << "]";
        return res;
    }

    unsigned int numberOfDimensions = actualShape.GetNumDimensions();

    if (!isDynamic)
    {
        // Checks they are same shape.
        for (unsigned int i = 0; i < numberOfDimensions; ++i)
        {
            if (actualShape[i] != expectedShape[i])
            {
                armnn::PredicateResult res(false);
                res.Message() << "Different shapes ["
                              << actualShape[i]
                              << "!="
                              << expectedShape[i]
                              << "]";
                return res;
            }
        }
    }

    // Fun iteration over n dimensions.
    std::vector<unsigned int> indices;
    for (unsigned int i = 0; i < numberOfDimensions; i++)
    {
        indices.emplace_back(0);
    }

    std::stringstream errorString;
    int numFailedElements = 0;
    constexpr int maxReportedDifferences = 3;
    unsigned int index = 0;

    // Compare data element by element.
    while (true)
    {
        bool comparison;
        // As true for uint8_t is non-zero (1-255) we must have a dedicated compare for Booleans.
        if(compareBoolean)
        {
            comparison = SelectiveCompareBoolean(actualData[index], expectedData[index]);
        }
        else
        {
            comparison = SelectiveCompare(actualData[index], expectedData[index]);
        }

        if (!comparison)
        {
            ++numFailedElements;

            if (numFailedElements <= maxReportedDifferences)
            {
                if (numFailedElements >= 2)
                {
                    errorString << ", ";
                }
                errorString << "[";
                for (unsigned int i = 0; i < numberOfDimensions; ++i)
                {
                    errorString << indices[i];
                    if (i != numberOfDimensions - 1)
                    {
                        errorString << ",";
                    }
                }
                errorString << "]";

                errorString << " (" << +actualData[index] << " != " << +expectedData[index] << ")";
            }
        }

        ++indices[numberOfDimensions - 1];
        for (unsigned int i=numberOfDimensions-1; i>0; i--)
        {
            if (indices[i] == actualShape[i])
            {
                indices[i] = 0;
                ++indices[i - 1];
            }
        }
        if (indices[0] == actualShape[0])
        {
            break;
        }

        index++;
    }

    armnn::PredicateResult comparisonResult(true);
    if (numFailedElements > 0)
    {
        comparisonResult.SetResult(false);
        comparisonResult.Message() << numFailedElements << " different values at: ";
        if (numFailedElements > maxReportedDifferences)
        {
            errorString << ", ... (and " << (numFailedElements - maxReportedDifferences) << " other differences)";
        }
        comparisonResult.Message() << errorString.str();
    }

    return comparisonResult;
}

template <typename T>
std::vector<T> MakeRandomTensor(const armnn::TensorInfo& tensorInfo,
                                unsigned int seed,
                                float        min = -10.0f,
                                float        max = 10.0f)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min, max);

    std::vector<float> init(tensorInfo.GetNumElements());
    for (unsigned int i = 0; i < init.size(); i++)
    {
        init[i] = dist(gen);
    }

    const float   qScale  = tensorInfo.GetQuantizationScale();
    const int32_t qOffset = tensorInfo.GetQuantizationOffset();

    return armnnUtils::QuantizedVector<T>(init, qScale, qOffset);
}
