//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>


#include <armnn/Utils.hpp>
#include <armnn/Types.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/Descriptors.hpp>
#include <GraphTopologicalSort.hpp>
#include <Graph.hpp>
#include <ResolveType.hpp>

BOOST_AUTO_TEST_SUITE(Utils)

BOOST_AUTO_TEST_CASE(DataTypeSize)
{
    BOOST_TEST(armnn::GetDataTypeSize(armnn::DataType::Float32) == 4);
    BOOST_TEST(armnn::GetDataTypeSize(armnn::DataType::QuantisedAsymm8) == 1);
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

BOOST_AUTO_TEST_SUITE_END()
