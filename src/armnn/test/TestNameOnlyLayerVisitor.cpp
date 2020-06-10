//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TestNameOnlyLayerVisitor.hpp"

#include <Network.hpp>

#include <boost/test/unit_test.hpp>

namespace
{

#define TEST_CASE_CHECK_LAYER_VISITOR_NAME(name) \
BOOST_AUTO_TEST_CASE(Check##name##LayerVisitorName) \
{ \
    Test##name##LayerVisitor visitor("name##Layer"); \
    armnn::Network net; \
    armnn::IConnectableLayer *const layer = net.Add##name##Layer("name##Layer"); \
    layer->Accept(visitor); \
}

#define TEST_CASE_CHECK_LAYER_VISITOR_NAME_NULLPTR(name) \
BOOST_AUTO_TEST_CASE(Check##name##LayerVisitorNameNullptr) \
{ \
    Test##name##LayerVisitor visitor; \
    armnn::Network net; \
    armnn::IConnectableLayer *const layer = net.Add##name##Layer(); \
    layer->Accept(visitor); \
}

#define TEST_SUITE_NAME_ONLY_LAYER_VISITOR(name) \
TEST_CASE_CHECK_LAYER_VISITOR_NAME(name) \
TEST_CASE_CHECK_LAYER_VISITOR_NAME_NULLPTR(name)

} // anonymous namespace

BOOST_AUTO_TEST_SUITE(TestNameOnlyLayerVisitor)

TEST_SUITE_NAME_ONLY_LAYER_VISITOR(Addition)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR(Dequantize)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR(Division)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR(Floor)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR(Maximum)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR(Merge)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR(Minimum)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR(Multiplication)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR(Prelu)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR(Quantize)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR(Rank)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR(Subtraction)
TEST_SUITE_NAME_ONLY_LAYER_VISITOR(Switch)

BOOST_AUTO_TEST_SUITE_END()
