//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Tensor.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnnUtils/FloatingPointComparison.hpp>

#include <QuantizeHelper.hpp>

#include <boost/multi_array.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/test/unit_test.hpp>

#include <array>
#include <cmath>
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

template <typename T, std::size_t n>
boost::test_tools::predicate_result CompareTensors(const boost::multi_array<T, n>& a,
                                                   const boost::multi_array<T, n>& b,
                                                   bool compareBoolean = false,
                                                   bool isDynamic = false)
{
    if (!isDynamic)
    {
        // Checks they are same shape.
        for (unsigned int i = 0;
             i < n;
             i++)
        {
            if (a.shape()[i] != b.shape()[i])
            {
                boost::test_tools::predicate_result res(false);
                res.message() << "Different shapes ["
                              << a.shape()[i]
                              << "!="
                              << b.shape()[i]
                              << "]";
                return res;
            }
        }
    }

    // Now compares element-wise.

    // Fun iteration over n dimensions.
    std::array<unsigned int, n> indices;
    for (unsigned int i = 0; i < n; i++)
    {
        indices[i] = 0;
    }

    std::stringstream errorString;
    int numFailedElements = 0;
    constexpr int maxReportedDifferences = 3;

    while (true)
    {
        bool comparison;
        // As true for uint8_t is non-zero (1-255) we must have a dedicated compare for Booleans.
        if(compareBoolean)
        {
            comparison = SelectiveCompareBoolean(a(indices), b(indices));
        }
        else
        {
            comparison = SelectiveCompare(a(indices), b(indices));
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
                for (unsigned int i = 0; i < n; ++i)
                {
                    errorString << indices[i];
                    if (i != n - 1)
                    {
                        errorString << ",";
                    }
                }
                errorString << "]";

                errorString << " (" << +a(indices) << " != " << +b(indices) << ")";
            }
        }

        ++indices[n - 1];
        for (unsigned int i=n-1; i>0; i--)
        {
            if (indices[i] == a.shape()[i])
            {
                indices[i] = 0;
                ++indices[i - 1];
            }
        }

        if (indices[0] == a.shape()[0])
        {
            break;
        }
    }

    boost::test_tools::predicate_result comparisonResult(true);
    if (numFailedElements > 0)
    {
        comparisonResult = false;
        comparisonResult.message() << numFailedElements << " different values at: ";
        if (numFailedElements > maxReportedDifferences)
        {
            errorString << ", ... (and " << (numFailedElements - maxReportedDifferences) << " other differences)";
        }
        comparisonResult.message() << errorString.str();
    }

    return comparisonResult;
}


// Creates a boost::multi_array with the shape defined by the given TensorInfo.
template <typename T, std::size_t n>
boost::multi_array<T, n> MakeTensor(const armnn::TensorInfo& tensorInfo)
{
    std::array<unsigned int, n> shape;

    for (unsigned int i = 0; i < n; i++)
    {
        shape[i] = tensorInfo.GetShape()[i];
    }

    return boost::multi_array<T, n>(shape);
}

// Creates a boost::multi_array with the shape defined by the given TensorInfo and contents defined by the given vector.
template <typename T, std::size_t n>
boost::multi_array<T, n> MakeTensor(
    const armnn::TensorInfo& tensorInfo, const std::vector<T>& flat, bool isDynamic = false)
{
    if (!isDynamic)
    {
        ARMNN_ASSERT_MSG(flat.size() == tensorInfo.GetNumElements(), "Wrong number of components supplied to tensor");
    }

    std::array<unsigned int, n> shape;

    // NOTE: tensorInfo.GetNumDimensions() might be different from n
    const unsigned int returnDimensions = static_cast<unsigned int>(n);
    const unsigned int actualDimensions = tensorInfo.GetNumDimensions();

    const unsigned int paddedDimensions =
        returnDimensions > actualDimensions ? returnDimensions - actualDimensions : 0u;

    for (unsigned int i = 0u; i < returnDimensions; i++)
    {
        if (i < paddedDimensions)
        {
            shape[i] = 1u;
        }
        else
        {
            shape[i] = tensorInfo.GetShape()[i - paddedDimensions];
        }
    }

    boost::const_multi_array_ref<T, n> arrayRef(&flat[0], shape);
    return boost::multi_array<T, n>(arrayRef);
}

template <typename T, std::size_t n>
boost::multi_array<T, n> MakeRandomTensor(const armnn::TensorInfo& tensorInfo,
                                          unsigned int seed,
                                          float        min = -10.0f,
                                          float        max = 10.0f)
{
    boost::random::mt19937                          gen(seed);
    boost::random::uniform_real_distribution<float> dist(min, max);

    std::vector<float> init(tensorInfo.GetNumElements());
    for (unsigned int i = 0; i < init.size(); i++)
    {
        init[i] = dist(gen);
    }

    const float   qScale  = tensorInfo.GetQuantizationScale();
    const int32_t qOffset = tensorInfo.GetQuantizationOffset();

    return MakeTensor<T, n>(tensorInfo, armnnUtils::QuantizedVector<T>(init, qScale, qOffset));
}
