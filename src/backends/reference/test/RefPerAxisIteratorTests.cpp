//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <reference/workloads/Decoders.hpp>

#include <fmt/format.h>

#include <doctest/doctest.h>

#include <chrono>

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

using namespace armnn;

// Basically a per axis decoder but without any decoding/quantization
class MockPerAxisIterator : public PerAxisIterator<const int8_t, Decoder<int8_t>>
{
public:
    MockPerAxisIterator(const int8_t* data, const armnn::TensorShape& tensorShape, const unsigned int axis)
            : PerAxisIterator(data, tensorShape, axis), m_NumElements(tensorShape.GetNumElements())
    {}

    int8_t Get() const override
    {
        return *m_Iterator;
    }

    virtual std::vector<float> DecodeTensor(const TensorShape &tensorShape,
                                            bool isDepthwise = false) override
    {
        IgnoreUnused(tensorShape, isDepthwise);
        return std::vector<float>{};
    };

    // Iterates over data using operator[] and returns vector
    std::vector<int8_t> Loop()
    {
        std::vector<int8_t> vec;
        for (uint32_t i = 0; i < m_NumElements; ++i)
        {
            this->operator[](i);
            vec.emplace_back(Get());
        }
        return vec;
    }

    unsigned int GetAxisIndex()
    {
        return m_AxisIndex;
    }
    unsigned int m_NumElements;
};

TEST_SUITE("RefPerAxisIterator")
{
// Test Loop (Equivalent to DecodeTensor) and Axis = 0
TEST_CASE("PerAxisIteratorTest1")
{
    std::vector<int8_t> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    TensorInfo tensorInfo ({3,1,2,2},DataType::QSymmS8);

    // test axis=0
    std::vector<int8_t> expOutput = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto iterator = MockPerAxisIterator(input.data(), tensorInfo.GetShape(), 0);
    std::vector<int8_t> output = iterator.Loop();
    CompareVector(output, expOutput);

    // Set iterator to index and check if the axis index is correct
    iterator[5];
    CHECK(iterator.GetAxisIndex() == 1u);

    iterator[1];
    CHECK(iterator.GetAxisIndex() == 0u);

    iterator[10];
    CHECK(iterator.GetAxisIndex() == 2u);
}

// Test Axis = 1
TEST_CASE("PerAxisIteratorTest2")
{
    std::vector<int8_t> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    TensorInfo tensorInfo ({3,1,2,2},DataType::QSymmS8);

    // test axis=1
    std::vector<int8_t> expOutput = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto iterator = MockPerAxisIterator(input.data(), tensorInfo.GetShape(), 1);
    std::vector<int8_t> output = iterator.Loop();
    CompareVector(output, expOutput);

    // Set iterator to index and check if the axis index is correct
    iterator[5];
    CHECK(iterator.GetAxisIndex() == 0u);

    iterator[1];
    CHECK(iterator.GetAxisIndex() == 0u);

    iterator[10];
    CHECK(iterator.GetAxisIndex() == 0u);
}

// Test Axis = 2
TEST_CASE("PerAxisIteratorTest3")
{
    std::vector<int8_t> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    TensorInfo tensorInfo ({3,1,2,2},DataType::QSymmS8);

    // test axis=2
    std::vector<int8_t> expOutput = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto iterator = MockPerAxisIterator(input.data(), tensorInfo.GetShape(), 2);
    std::vector<int8_t> output = iterator.Loop();
    CompareVector(output, expOutput);

    // Set iterator to index and check if the axis index is correct
    iterator[5];
    CHECK(iterator.GetAxisIndex() == 0u);

    iterator[1];
    CHECK(iterator.GetAxisIndex() == 0u);

    iterator[10];
    CHECK(iterator.GetAxisIndex() == 1u);
}

// Test Axis = 3
TEST_CASE("PerAxisIteratorTest4")
{
    std::vector<int8_t> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    TensorInfo tensorInfo ({3,1,2,2},DataType::QSymmS8);

    // test axis=3
    std::vector<int8_t> expOutput = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto iterator = MockPerAxisIterator(input.data(), tensorInfo.GetShape(), 3);
    std::vector<int8_t> output = iterator.Loop();
    CompareVector(output, expOutput);

    // Set iterator to index and check if the axis index is correct
    iterator[5];
    CHECK(iterator.GetAxisIndex() == 1u);

    iterator[1];
    CHECK(iterator.GetAxisIndex() == 1u);

    iterator[10];
    CHECK(iterator.GetAxisIndex() == 0u);
}

// Test Axis = 1. Different tensor shape
TEST_CASE("PerAxisIteratorTest5")
{
    using namespace armnn;
    std::vector<int8_t> input =
    {
         0,  1,  2,  3,
         4,  5,  6,  7,
         8,  9, 10, 11,
        12, 13, 14, 15
    };

    std::vector<int8_t> expOutput =
    {
         0,  1,  2,  3,
         4,  5,  6,  7,
         8,  9, 10, 11,
        12, 13, 14, 15
    };

    TensorInfo tensorInfo ({2,2,2,2},DataType::QSymmS8);
    auto iterator = MockPerAxisIterator(input.data(), tensorInfo.GetShape(), 1);
    std::vector<int8_t> output = iterator.Loop();
    CompareVector(output, expOutput);

    // Set iterator to index and check if the axis index is correct
    iterator[5];
    CHECK(iterator.GetAxisIndex() == 1u);

    iterator[1];
    CHECK(iterator.GetAxisIndex() == 0u);

    iterator[10];
    CHECK(iterator.GetAxisIndex() == 0u);
}

// Test the increment and decrement operator
TEST_CASE("PerAxisIteratorTest7")
{
    using namespace armnn;
    std::vector<int8_t> input =
    {
        0, 1,  2,  3,
        4, 5,  6,  7,
        8, 9, 10, 11
    };

    std::vector<int8_t> expOutput =
    {
        0, 1,  2,  3,
        4, 5,  6,  7,
        8, 9, 10, 11
    };

    TensorInfo tensorInfo ({3,1,2,2},DataType::QSymmS8);
    auto iterator = MockPerAxisIterator(input.data(), tensorInfo.GetShape(), 2);

    iterator += 3;
    CHECK(iterator.Get() == expOutput[3]);
    CHECK(iterator.GetAxisIndex() == 1u);

    iterator += 3;
    CHECK(iterator.Get() == expOutput[6]);
    CHECK(iterator.GetAxisIndex() == 1u);

    iterator -= 2;
    CHECK(iterator.Get() == expOutput[4]);
    CHECK(iterator.GetAxisIndex() == 0u);

    iterator -= 1;
    CHECK(iterator.Get() == expOutput[3]);
    CHECK(iterator.GetAxisIndex() == 1u);
}

}