//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TestNameOnlyLayerVisitor.hpp"

#include <Network.hpp>

#include <boost/test/unit_test.hpp>

namespace
{

#define ADD_LAYER_METHOD_1_PARAM(name) net.Add##name##Layer("name##Layer")
#define ADD_LAYER_METHOD_2_PARAM(name) net.Add##name##Layer(armnn::name##Descriptor(), "name##Layer")

#define TEST_CASE_CHECK_LAYER_VISITOR_NAME(name, numParams) \
BOOST_AUTO_TEST_CASE(Check##name##LayerVisitorName) \
{ \
    Test##name##LayerVisitor visitor("name##Layer"); \
    armnn::Network net; \
    armnn::IConnectableLayer *const layer = ADD_LAYER_METHOD_##numParams##_PARAM(name); \
    layer->Accept(visitor); \
}

#define ADD_LAYER_METHOD_NULLPTR_1_PARAM(name) net.Add##name##Layer()
#define ADD_LAYER_METHOD_NULLPTR_2_PARAM(name) net.Add##name##Layer(armnn::name##Descriptor())

#define TEST_CASE_CHECK_LAYER_VISITOR_NAME_NULLPTR(name, numParams) \
BOOST_AUTO_TEST_CASE(Check##name##LayerVisitorNameNullptr) \
{ \
    Test##name##LayerVisitor visitor; \
    armnn::Network net; \
    armnn::IConnectableLayer *const layer = ADD_LAYER_METHOD_NULLPTR_##numParams##_PARAM(name); \
    layer->Accept(visitor); \
}

#define TEST_SUITE_NAME_ONLY_LAYER_VISITOR_1_PARAM(name) \
TEST_CASE_CHECK_LAYER_VISITOR_NAME(name, 1) \
TEST_CASE_CHECK_LAYER_VISITOR_NAME_NULLPTR(name, 1)

#define TEST_SUITE_NAME_ONLY_LAYER_VISITOR_2_PARAM(name) \
TEST_CASE_CHECK_LAYER_VISITOR_NAME(name, 2) \
TEST_CASE_CHECK_LAYER_VISITOR_NAME_NULLPTR(name, 2)

} // anonymous namespace

BOOST_AUTO_TEST_SUITE(TestNameOnlyLayerVisitor)

TEST_SUITE_NAME_ONLY_LAYER_VISITOR_1_PARAM(Addition)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR_2_PARAM(DepthToSpace)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR_1_PARAM(Division)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR_1_PARAM(Equal)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR_1_PARAM(Floor)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR_1_PARAM(Gather)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR_1_PARAM(Greater)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR_1_PARAM(Maximum)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR_1_PARAM(Minimum)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR_1_PARAM(Multiplication)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR_1_PARAM(Rsqrt)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR_2_PARAM(Slice)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR_1_PARAM(Subtraction)

BOOST_AUTO_TEST_SUITE_END()
