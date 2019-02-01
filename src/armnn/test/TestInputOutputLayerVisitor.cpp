//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "TestInputOutputLayerVisitor.hpp"
#include "Network.hpp"

namespace armnn
{

BOOST_AUTO_TEST_SUITE(TestInputOutputLayerVisitor)

BOOST_AUTO_TEST_CASE(CheckInputLayerVisitorBindingIdAndName)
{
    const char* layerName = "InputLayer";
    TestInputLayerVisitor visitor(1, layerName);
    Network net;

    IConnectableLayer *const layer = net.AddInputLayer(1, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckInputLayerVisitorBindingIdAndNameNull)
{
    TestInputLayerVisitor visitor(1);
    Network net;

    IConnectableLayer *const layer = net.AddInputLayer(1);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckOutputLayerVisitorBindingIdAndName)
{
    const char* layerName = "OutputLayer";
    TestOutputLayerVisitor visitor(1, layerName);
    Network net;

    IConnectableLayer *const layer = net.AddOutputLayer(1, layerName);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_CASE(CheckOutputLayerVisitorBindingIdAndNameNull)
{
    TestOutputLayerVisitor visitor(1);
    Network net;

    IConnectableLayer *const layer = net.AddOutputLayer(1);
    layer->Accept(visitor);
}

BOOST_AUTO_TEST_SUITE_END()

} //namespace armnn