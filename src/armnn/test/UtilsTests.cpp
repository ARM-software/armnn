//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>


#include <armnn/Utils.hpp>
#include <armnn/Types.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/Descriptors.hpp>
#include <armnnUtils/Permute.hpp>
#include <GraphTopologicalSort.hpp>
#include <Graph.hpp>
#include <ResolveType.hpp>

BOOST_AUTO_TEST_SUITE(Utils)

BOOST_AUTO_TEST_CASE(DataTypeSize)
{
    BOOST_TEST(armnn::GetDataTypeSize(armnn::DataType::Float32) == 4);
    BOOST_TEST(armnn::GetDataTypeSize(armnn::DataType::QAsymmU8) == 1);
    BOOST_TEST(armnn::GetDataTypeSize(armnn::DataType::Signed32) == 4);
    BOOST_TEST(armnn::GetDataTypeSize(armnn::DataType::Boolean) == 1);
}

BOOST_AUTO_TEST_CASE(PermuteDescriptorWithTooManyMappings)
{
    BOOST_CHECK_THROW(armnn::PermuteDescriptor({ 0u, 1u, 2u, 3u, 4u, 5u }), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(PermuteDescriptorWithInvalidMappings1d)
{
    BOOST_CHECK_THROW(armnn::PermuteDescriptor({ 1u }), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(PermuteDescriptorWithInvalidMappings2d)
{
    BOOST_CHECK_THROW(armnn::PermuteDescriptor({ 2u, 0u }), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(PermuteDescriptorWithInvalidMappings3d)
{
    BOOST_CHECK_THROW(armnn::PermuteDescriptor({ 0u, 3u, 1u }), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(PermuteDescriptorWithInvalidMappings4d)
{
    BOOST_CHECK_THROW(armnn::PermuteDescriptor({ 0u, 1u, 2u, 4u }), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(PermuteDescriptorWithInvalidMappings5d)
{
    BOOST_CHECK_THROW(armnn::PermuteDescriptor({ 0u, 1u, 2u, 3u, 5u }), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(PermuteDescriptorWithDuplicatedMappings)
{
    BOOST_CHECK_THROW(armnn::PermuteDescriptor({ 1u, 1u, 0u }), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(HalfType)
{
    using namespace half_float::literal;
    armnn::Half a = 1.0_h;

    float b = 1.0f;
    armnn::Half c(b);

    // Test half type
    BOOST_CHECK_EQUAL(a, b);
    BOOST_CHECK_EQUAL(sizeof(c), 2);

    // Test half type is floating point type
    BOOST_CHECK(std::is_floating_point<armnn::Half>::value);

    // Test utility function returns correct type.
    using ResolvedType = armnn::ResolveType<armnn::DataType::Float16>;
    constexpr bool isHalfType = std::is_same<armnn::Half, ResolvedType>::value;
    BOOST_CHECK(isHalfType);

    //Test utility functions return correct size
    BOOST_CHECK(GetDataTypeSize(armnn::DataType::Float16) == 2);

    //Test utility functions return correct name
    BOOST_CHECK((GetDataTypeName(armnn::DataType::Float16) == std::string("Float16")));
}

BOOST_AUTO_TEST_CASE(BFloatType)
{
    uint16_t v = 16256;
    armnn::BFloat16 a(v);
    armnn::BFloat16 b(1.0f);
    armnn::BFloat16 zero;

    // Test BFloat16 type
    BOOST_CHECK_EQUAL(sizeof(a), 2);
    BOOST_CHECK_EQUAL(a, b);
    BOOST_CHECK_EQUAL(a.Val(), v);
    BOOST_CHECK_EQUAL(a, 1.0f);
    BOOST_CHECK_EQUAL(zero, 0.0f);

    // Infinity
    float infFloat = std::numeric_limits<float>::infinity();
    armnn::BFloat16 infBF(infFloat);
    BOOST_CHECK_EQUAL(infBF, armnn::BFloat16::Inf());

    // NaN
    float nan = std::numeric_limits<float>::quiet_NaN();
    armnn::BFloat16 nanBF(nan);
    BOOST_CHECK_EQUAL(nanBF, armnn::BFloat16::Nan());

    // Test utility function returns correct type.
    using ResolvedType = armnn::ResolveType<armnn::DataType::BFloat16>;
    constexpr bool isBFloat16Type = std::is_same<armnn::BFloat16, ResolvedType>::value;
    BOOST_CHECK(isBFloat16Type);

    //Test utility functions return correct size
    BOOST_CHECK(GetDataTypeSize(armnn::DataType::BFloat16) == 2);

    //Test utility functions return correct name
    BOOST_CHECK((GetDataTypeName(armnn::DataType::BFloat16) == std::string("BFloat16")));
}

BOOST_AUTO_TEST_CASE(Float32ToBFloat16Test)
{
    // LSB = 0, R = 0 -> round down
    armnn::BFloat16 roundDown0 = armnn::BFloat16::Float32ToBFloat16(1.704735E38f); // 0x7F004000
    BOOST_CHECK_EQUAL(roundDown0.Val(), 0x7F00);
    // LSB = 1, R = 0 -> round down
    armnn::BFloat16 roundDown1 = armnn::BFloat16::Float32ToBFloat16(9.18355E-41f); // 0x00010000
    BOOST_CHECK_EQUAL(roundDown1.Val(), 0x0001);
    // LSB = 0, R = 1 all 0 -> round down
    armnn::BFloat16 roundDown2 = armnn::BFloat16::Float32ToBFloat16(1.14794E-40f); // 0x00014000
    BOOST_CHECK_EQUAL(roundDown2.Val(), 0x0001);
    // LSB = 1, R = 1 -> round up
    armnn::BFloat16 roundUp = armnn::BFloat16::Float32ToBFloat16(-2.0234377f); // 0xC0018001
    BOOST_CHECK_EQUAL(roundUp.Val(), 0xC002);
    // LSB = 0, R = 1 -> round up
    armnn::BFloat16 roundUp1 = armnn::BFloat16::Float32ToBFloat16(4.843037E-35f); // 0x0680C000
    BOOST_CHECK_EQUAL(roundUp1.Val(), 0x0681);
    // Max positive value -> infinity
    armnn::BFloat16 maxPositive = armnn::BFloat16::Float32ToBFloat16(std::numeric_limits<float>::max()); // 0x7F7FFFFF
    BOOST_CHECK_EQUAL(maxPositive, armnn::BFloat16::Inf());
    // Max negative value -> -infinity
    armnn::BFloat16 maxNeg = armnn::BFloat16::Float32ToBFloat16(std::numeric_limits<float>::lowest()); // 0xFF7FFFFF
    BOOST_CHECK_EQUAL(maxNeg.Val(), 0xFF80);
    // Min positive value
    armnn::BFloat16 minPositive = armnn::BFloat16::Float32ToBFloat16(1.1754942E-38f); // 0x007FFFFF
    BOOST_CHECK_EQUAL(minPositive.Val(), 0x0080);
    // Min negative value
    armnn::BFloat16 minNeg = armnn::BFloat16::Float32ToBFloat16(-1.1754942E-38f); // 0x807FFFFF
    BOOST_CHECK_EQUAL(minNeg.Val(), 0x8080);
}

BOOST_AUTO_TEST_CASE(BFloat16ToFloat32Test)
{
    armnn::BFloat16 bf0(1.5f);
    BOOST_CHECK_EQUAL(bf0.ToFloat32(), 1.5f);
    armnn::BFloat16 bf1(-5.525308E-25f);
    BOOST_CHECK_EQUAL(bf1.ToFloat32(), -5.525308E-25f);
    armnn::BFloat16 bf2(-2.0625f);
    BOOST_CHECK_EQUAL(bf2.ToFloat32(), -2.0625f);
    uint16_t v = 32639;
    armnn::BFloat16 bf3(v);
    BOOST_CHECK_EQUAL(bf3.ToFloat32(), 3.3895314E38f);
    // Infinity
    BOOST_CHECK_EQUAL(armnn::BFloat16::Inf().ToFloat32(), std::numeric_limits<float>::infinity());
    // NaN
    BOOST_CHECK(std::isnan(armnn::BFloat16::Nan().ToFloat32()));
}

BOOST_AUTO_TEST_CASE(GraphTopologicalSortSimpleTest)
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

    BOOST_TEST(sortCompleted);

    std::vector<int> correctResult = {5, 4, 2, 0, 3, 1};
    BOOST_CHECK_EQUAL_COLLECTIONS(output.begin(), output.end(), correctResult.begin(), correctResult.end());
}

BOOST_AUTO_TEST_CASE(GraphTopologicalSortVariantTest)
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

    BOOST_TEST(sortCompleted);

    std::vector<int> correctResult = {6, 5, 3, 4, 2, 0, 1};
    BOOST_CHECK_EQUAL_COLLECTIONS(output.begin(), output.end(), correctResult.begin(), correctResult.end());
}

BOOST_AUTO_TEST_CASE(CyclicalGraphTopologicalSortTest)
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

    BOOST_TEST(!sortCompleted);
}

BOOST_AUTO_TEST_CASE(PermuteQuantizationDim)
{
    std::vector<float> scales;

    // Set QuantizationDim to be index 1
    const armnn::TensorInfo info({ 1, 2, 3, 4 }, armnn::DataType::Float32, scales, 1U);
    BOOST_CHECK(info.GetQuantizationDim().value() == 1U);

    // Permute so that index 1 moves to final index i.e. index 3
    armnn::PermutationVector mappings({ 0, 3, 2, 1 });
    auto permutedPerChannel = armnnUtils::Permuted(info, mappings, true);
    auto permuted = armnnUtils::Permuted(info, mappings);

    // Check that QuantizationDim is in index 3
    BOOST_CHECK(permutedPerChannel.GetQuantizationDim().value() == 3U);

    // Check previous implementation unchanged
    BOOST_CHECK(permuted.GetQuantizationDim().value() == 1U);
}

BOOST_AUTO_TEST_SUITE_END()
