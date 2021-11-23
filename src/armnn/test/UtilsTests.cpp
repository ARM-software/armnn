//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <doctest/doctest.h>


#include <armnn/BackendHelper.hpp>
#include <armnn/Utils.hpp>
#include <armnn/Types.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/Descriptors.hpp>
#include <armnnUtils/Permute.hpp>
#include <GraphTopologicalSort.hpp>
#include <Graph.hpp>
#include <ResolveType.hpp>

TEST_SUITE("Utils")
{
TEST_CASE("DataTypeSize")
{
    CHECK(armnn::GetDataTypeSize(armnn::DataType::Float32) == 4);
    CHECK(armnn::GetDataTypeSize(armnn::DataType::QAsymmU8) == 1);
    CHECK(armnn::GetDataTypeSize(armnn::DataType::Signed32) == 4);
    CHECK(armnn::GetDataTypeSize(armnn::DataType::Boolean) == 1);
}

TEST_CASE("PermuteDescriptorWithTooManyMappings")
{
    CHECK_THROWS_AS(armnn::PermuteDescriptor({ 0u, 1u, 2u, 3u, 4u, 5u }), armnn::InvalidArgumentException);
}

TEST_CASE("PermuteDescriptorWithInvalidMappings1d")
{
    CHECK_THROWS_AS(armnn::PermuteDescriptor({ 1u }), armnn::InvalidArgumentException);
}

TEST_CASE("PermuteDescriptorWithInvalidMappings2d")
{
    CHECK_THROWS_AS(armnn::PermuteDescriptor({ 2u, 0u }), armnn::InvalidArgumentException);
}

TEST_CASE("PermuteDescriptorWithInvalidMappings3d")
{
    CHECK_THROWS_AS(armnn::PermuteDescriptor({ 0u, 3u, 1u }), armnn::InvalidArgumentException);
}

TEST_CASE("PermuteDescriptorWithInvalidMappings4d")
{
    CHECK_THROWS_AS(armnn::PermuteDescriptor({ 0u, 1u, 2u, 4u }), armnn::InvalidArgumentException);
}

TEST_CASE("PermuteDescriptorWithInvalidMappings5d")
{
    CHECK_THROWS_AS(armnn::PermuteDescriptor({ 0u, 1u, 2u, 3u, 5u }), armnn::InvalidArgumentException);
}

TEST_CASE("PermuteDescriptorWithDuplicatedMappings")
{
    CHECK_THROWS_AS(armnn::PermuteDescriptor({ 1u, 1u, 0u }), armnn::InvalidArgumentException);
}

TEST_CASE("HalfType")
{
    using namespace half_float::literal;
    armnn::Half a = 1.0_h;

    float b = 1.0f;
    armnn::Half c(b);

    // Test half type
    CHECK_EQ(a, b);
    CHECK_EQ(sizeof(c), 2);

    // Test half type is floating point type
    CHECK(std::is_floating_point<armnn::Half>::value);

    // Test utility function returns correct type.
    using ResolvedType = armnn::ResolveType<armnn::DataType::Float16>;
    constexpr bool isHalfType = std::is_same<armnn::Half, ResolvedType>::value;
    CHECK(isHalfType);

    //Test utility functions return correct size
    CHECK(GetDataTypeSize(armnn::DataType::Float16) == 2);

    //Test utility functions return correct name
    CHECK((GetDataTypeName(armnn::DataType::Float16) == std::string("Float16")));
}

TEST_CASE("BFloatType")
{
    uint16_t v = 16256;
    armnn::BFloat16 a(v);
    armnn::BFloat16 b(1.0f);
    armnn::BFloat16 zero;

    // Test BFloat16 type
    CHECK_EQ(sizeof(a), 2);
    CHECK_EQ(a, b);
    CHECK_EQ(a.Val(), v);
    CHECK_EQ(a, 1.0f);
    CHECK_EQ(zero, 0.0f);

    // Infinity
    float infFloat = std::numeric_limits<float>::infinity();
    armnn::BFloat16 infBF(infFloat);
    CHECK_EQ(infBF, armnn::BFloat16::Inf());

    // NaN
    float nan = std::numeric_limits<float>::quiet_NaN();
    armnn::BFloat16 nanBF(nan);
    CHECK_EQ(nanBF, armnn::BFloat16::Nan());

    // Test utility function returns correct type.
    using ResolvedType = armnn::ResolveType<armnn::DataType::BFloat16>;
    constexpr bool isBFloat16Type = std::is_same<armnn::BFloat16, ResolvedType>::value;
    CHECK(isBFloat16Type);

    //Test utility functions return correct size
    CHECK(GetDataTypeSize(armnn::DataType::BFloat16) == 2);

    //Test utility functions return correct name
    CHECK((GetDataTypeName(armnn::DataType::BFloat16) == std::string("BFloat16")));
}

TEST_CASE("Float32ToBFloat16Test")
{
    // LSB = 0, R = 0 -> round down
    armnn::BFloat16 roundDown0 = armnn::BFloat16::Float32ToBFloat16(1.704735E38f); // 0x7F004000
    CHECK_EQ(roundDown0.Val(), 0x7F00);
    // LSB = 1, R = 0 -> round down
    armnn::BFloat16 roundDown1 = armnn::BFloat16::Float32ToBFloat16(9.18355E-41f); // 0x00010000
    CHECK_EQ(roundDown1.Val(), 0x0001);
    // LSB = 0, R = 1 all 0 -> round down
    armnn::BFloat16 roundDown2 = armnn::BFloat16::Float32ToBFloat16(1.14794E-40f); // 0x00014000
    CHECK_EQ(roundDown2.Val(), 0x0001);
    // LSB = 1, R = 1 -> round up
    armnn::BFloat16 roundUp = armnn::BFloat16::Float32ToBFloat16(-2.0234377f); // 0xC0018001
    CHECK_EQ(roundUp.Val(), 0xC002);
    // LSB = 0, R = 1 -> round up
    armnn::BFloat16 roundUp1 = armnn::BFloat16::Float32ToBFloat16(4.843037E-35f); // 0x0680C000
    CHECK_EQ(roundUp1.Val(), 0x0681);
    // Max positive value -> infinity
    armnn::BFloat16 maxPositive = armnn::BFloat16::Float32ToBFloat16(std::numeric_limits<float>::max()); // 0x7F7FFFFF
    CHECK_EQ(maxPositive, armnn::BFloat16::Inf());
    // Max negative value -> -infinity
    armnn::BFloat16 maxNeg = armnn::BFloat16::Float32ToBFloat16(std::numeric_limits<float>::lowest()); // 0xFF7FFFFF
    CHECK_EQ(maxNeg.Val(), 0xFF80);
    // Min positive value
    armnn::BFloat16 minPositive = armnn::BFloat16::Float32ToBFloat16(1.1754942E-38f); // 0x007FFFFF
    CHECK_EQ(minPositive.Val(), 0x0080);
    // Min negative value
    armnn::BFloat16 minNeg = armnn::BFloat16::Float32ToBFloat16(-1.1754942E-38f); // 0x807FFFFF
    CHECK_EQ(minNeg.Val(), 0x8080);
}

TEST_CASE("BFloat16ToFloat32Test")
{
    armnn::BFloat16 bf0(1.5f);
    CHECK_EQ(bf0.ToFloat32(), 1.5f);
    armnn::BFloat16 bf1(-5.525308E-25f);
    CHECK_EQ(bf1.ToFloat32(), -5.525308E-25f);
    armnn::BFloat16 bf2(-2.0625f);
    CHECK_EQ(bf2.ToFloat32(), -2.0625f);
    uint16_t v = 32639;
    armnn::BFloat16 bf3(v);
    CHECK_EQ(bf3.ToFloat32(), 3.3895314E38f);
    // Infinity
    CHECK_EQ(armnn::BFloat16::Inf().ToFloat32(), std::numeric_limits<float>::infinity());
    // NaN
    CHECK(std::isnan(armnn::BFloat16::Nan().ToFloat32()));
}

TEST_CASE("GraphTopologicalSortSimpleTest")
{
    std::map<int, std::vector<int>> graph;

    graph[0] = {2};
    graph[1] = {3};
    graph[2] = {4};
    graph[3] = {4};
    graph[4] = {5};
    graph[5] = {};

    auto getNodeInputs = [graph](int node) -> std::vector<int>
    {
        return graph.find(node)->second;
    };

    std::vector<int> targetNodes = {0, 1};

    std::vector<int> output;
    bool sortCompleted = armnnUtils::GraphTopologicalSort<int>(targetNodes, getNodeInputs, output);

    CHECK(sortCompleted);

    std::vector<int> correctResult = {5, 4, 2, 0, 3, 1};
    CHECK(std::equal(output.begin(), output.end(), correctResult.begin(), correctResult.end()));
}

TEST_CASE("GraphTopologicalSortVariantTest")
{
    std::map<int, std::vector<int>> graph;

    graph[0] = {2};
    graph[1] = {2};
    graph[2] = {3, 4};
    graph[3] = {5};
    graph[4] = {5};
    graph[5] = {6};
    graph[6] = {};

    auto getNodeInputs = [graph](int node) -> std::vector<int>
    {
        return graph.find(node)->second;
    };

    std::vector<int> targetNodes = {0, 1};

    std::vector<int> output;
    bool sortCompleted = armnnUtils::GraphTopologicalSort<int>(targetNodes, getNodeInputs, output);

    CHECK(sortCompleted);

    std::vector<int> correctResult = {6, 5, 3, 4, 2, 0, 1};
    CHECK(std::equal(output.begin(), output.end(), correctResult.begin(), correctResult.end()));
}

TEST_CASE("CyclicalGraphTopologicalSortTest")
{
    std::map<int, std::vector<int>> graph;

    graph[0] = {1};
    graph[1] = {2};
    graph[2] = {0};

    auto getNodeInputs = [graph](int node) -> std::vector<int>
    {
        return graph.find(node)->second;
    };

    std::vector<int> targetNodes = {0};

    std::vector<int> output;
    bool sortCompleted = armnnUtils::GraphTopologicalSort<int>(targetNodes, getNodeInputs, output);

    CHECK(!sortCompleted);
}

TEST_CASE("PermuteQuantizationDim")
{
    std::vector<float> scales {1.0f, 1.0f};

    // Set QuantizationDim to be index 1
    const armnn::TensorInfo perChannelInfo({ 1, 2, 3, 4 }, armnn::DataType::Float32, scales, 1U);
    CHECK(perChannelInfo.GetQuantizationDim().value() == 1U);

    // Permute so that index 1 moves to final index i.e. index 3
    armnn::PermutationVector mappings({ 0, 3, 2, 1 });
    auto permutedPerChannel = armnnUtils::Permuted(perChannelInfo, mappings);

    // Check that QuantizationDim is in index 3
    CHECK(permutedPerChannel.GetQuantizationDim().value() == 3U);

    // Even if there is only a single scale the quantization dim still exists and needs to be permuted
    std::vector<float> scale {1.0f};
    const armnn::TensorInfo perChannelInfo1({ 1, 2, 3, 4 }, armnn::DataType::Float32, scale, 1U);
    auto permuted = armnnUtils::Permuted(perChannelInfo1, mappings);
    CHECK(permuted.GetQuantizationDim().value() == 3U);
}

TEST_CASE("EmptyPermuteVectorIndexOutOfBounds")
{
    armnn::PermutationVector pv = armnn::PermutationVector({});
    CHECK_THROWS_AS(pv[0], armnn::InvalidArgumentException);
}

TEST_CASE("PermuteDescriptorIndexOutOfBounds")
{
    armnn::PermutationVector pv = armnn::PermutationVector({ 1u, 2u, 0u });
    armnn::PermuteDescriptor desc = armnn::PermuteDescriptor(pv);
    CHECK_THROWS_AS(desc.m_DimMappings[3], armnn::InvalidArgumentException);
    CHECK(desc.m_DimMappings[0] == 1u);
}

TEST_CASE("TransposeDescriptorIndexOutOfBounds")
{
    armnn::PermutationVector pv = armnn::PermutationVector({ 2u, 1u, 0u });
    armnn::TransposeDescriptor desc = armnn::TransposeDescriptor(pv);
    CHECK_THROWS_AS(desc.m_DimMappings[3], armnn::InvalidArgumentException);
    CHECK(desc.m_DimMappings[2] == 0u);
}

TEST_CASE("PermuteVectorIterator")
{
    // We're slightly breaking the spirit of std::array.end() because we're using it as a
    // variable length rather than fixed length. This test is to use a couple of iterators and
    // make sure it still mostly makes sense.

    // Create zero length.
    armnn::PermutationVector zeroPVector({});
    // Begin should be equal to end.
    CHECK(zeroPVector.begin() == zeroPVector.end());

    // Create length 4. Summing the 4 values should be 6.
    armnn::PermutationVector fourPVector({ 0, 3, 2, 1 });
    unsigned int sum = 0;
    for (unsigned int it : fourPVector)
    {
        sum += it;
    }
    CHECK(sum == 6);
    // Directly use begin and end, make sure there are 4 iterations.
    unsigned int iterations = 0;
    auto itr = fourPVector.begin();
    while(itr != fourPVector.end())
    {
        ++iterations;
        itr++;
    }
    CHECK(iterations == 4);

    // Do the same with 2 elements.
    armnn::PermutationVector twoPVector({ 0, 1 });
    iterations = 0;
    itr = twoPVector.begin();
    while(itr != twoPVector.end())
    {
        ++iterations;
        itr++;
    }
    CHECK(iterations == 2);
}

#if defined(ARMNNREF_ENABLED)
TEST_CASE("LayerSupportHandle")
{
    auto layerSupportObject = armnn::GetILayerSupportByBackendId("CpuRef");
    armnn::TensorInfo input;
    std::string reasonIfUnsupported;
    // InputLayer always supported for CpuRef
    CHECK_EQ(layerSupportObject.IsInputSupported(input, reasonIfUnsupported), true);

    CHECK(layerSupportObject.IsBackendRegistered());
}
#endif

}
