//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>
#include "TestNameOnlyLayerVisitor.hpp"
#include "Network.hpp"

namespace armnn {

BOOST_AUTO_TEST_SUITE(TestNameOnlyLayerVisitor)

BOOST_AUTO_TEST_CASE(CheckAdditionLayerVisitorName)
{
    TestAdditionLayerVisitor visitor("AdditionLayer");
    Network net;

    IConnectableLayer *const layer = net.AddAdditionLayer("AdditionLayer");
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckAdditionLayerVisitorNameNullptr)
{
    TestAdditionLayerVisitor visitor;
    Network net;

    IConnectableLayer *const layer = net.AddAdditionLayer();
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckMultiplicationLayerVisitorName)
{
    TestMultiplicationLayerVisitor visitor("MultiplicationLayer");
    Network net;

    IConnectableLayer *const layer = net.AddMultiplicationLayer("MultiplicationLayer");
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckMultiplicationLayerVisitorNameNullptr)
{
    TestMultiplicationLayerVisitor visitor;
    Network net;

    IConnectableLayer *const layer = net.AddMultiplicationLayer();
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckFloorLayerVisitorName)
{
    TestFloorLayerVisitor visitor("FloorLayer");
    Network net;

    IConnectableLayer *const layer = net.AddFloorLayer("FloorLayer");
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckFloorLayerVisitorNameNullptr)
{
    TestFloorLayerVisitor visitor;
    Network net;

    IConnectableLayer *const layer = net.AddFloorLayer();
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckDivisionLayerVisitorName)
{
    TestDivisionLayerVisitor visitor("DivisionLayer");
    Network net;

    IConnectableLayer *const layer = net.AddAdditionLayer("DivisionLayer");
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckDivisionLayerVisitorNameNullptr)
{
    TestDivisionLayerVisitor visitor;
    Network net;

    IConnectableLayer *const layer = net.AddDivisionLayer();
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckSubtractionLayerVisitorName)
{
    TestSubtractionLayerVisitor visitor("SubtractionLayer");
    Network net;

    IConnectableLayer *const layer = net.AddSubtractionLayer("SubtractionLayer");
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckSubtractionLayerVisitorNameNullptr)
{
    TestSubtractionLayerVisitor visitor;
    Network net;

    IConnectableLayer *const layer = net.AddSubtractionLayer();
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckMaximumLayerVisitorName)
{
    TestMaximumLayerVisitor visitor("MaximumLayer");
    Network net;

    IConnectableLayer *const layer = net.AddMaximumLayer("MaximumLayer");
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckMaximumLayerVisitorNameNullptr)
{
    TestMaximumLayerVisitor visitor;
    Network net;

    IConnectableLayer *const layer = net.AddMaximumLayer();
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckMinimumLayerVisitorName)
{
    TestMinimumLayerVisitor visitor("MinimumLayer");
    Network net;

    IConnectableLayer *const layer = net.AddMinimumLayer("MinimumLayer");
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckMinimumLayerVisitorNameNullptr)
{
    TestMinimumLayerVisitor visitor;
    Network net;

    IConnectableLayer *const layer = net.AddMinimumLayer();
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckGreaterLayerVisitorName)
{
    TestGreaterLayerVisitor visitor("GreaterLayer");
    Network net;

    IConnectableLayer *const layer = net.AddGreaterLayer("GreaterLayer");
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckGreaterLayerVisitorNameNullptr)
{
    TestGreaterLayerVisitor visitor;
    Network net;

    IConnectableLayer *const layer = net.AddGreaterLayer();
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckEqualLayerVisitorName)
{
    TestEqualLayerVisitor visitor("EqualLayer");
    Network net;

    IConnectableLayer *const layer = net.AddEqualLayer("EqualLayer");
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckEqualLayerVisitorNameNullptr)
{
    TestEqualLayerVisitor visitor;
    Network net;

    IConnectableLayer *const layer = net.AddEqualLayer();
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckRsqrtLayerVisitorName)
{
    TestRsqrtLayerVisitor visitor("RsqrtLayer");
    Network net;

    IConnectableLayer *const layer = net.AddRsqrtLayer("RsqrtLayer");
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckRsqrtLayerVisitorNameNullptr)
{
    TestRsqrtLayerVisitor visitor;
    Network net;

    IConnectableLayer *const layer = net.AddRsqrtLayer();
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckGatherLayerVisitorName)
{
    TestGatherLayerVisitor visitor("GatherLayer");
    Network net;

    IConnectableLayer *const layer = net.AddGatherLayer("GatherLayer");
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckGatherLayerVisitorNameNullptr)
{
    TestGatherLayerVisitor visitor;
    Network net;

    IConnectableLayer *const layer = net.AddGatherLayer();
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_SUITE_END()

} //namespace armnn