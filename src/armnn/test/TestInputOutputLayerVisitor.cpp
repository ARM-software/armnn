//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "TestInputOutputLayerVisitor.hpp"
#include "Network.hpp"

#include <doctest/doctest.h>

namespace armnn
{

TEST_SUITE("TestInputOutputLayerVisitor")
{
TEST_CASE("CheckInputLayerVisitorBindingIdAndName")
{
    const char* layerName = "InputLayer";
    TestInputLayerVisitor visitor(1, layerName);
    NetworkImpl net;

    IConnectableLayer *const layer = net.AddInputLayer(1, layerName);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckInputLayerVisitorBindingIdAndNameNull")
{
    TestInputLayerVisitor visitor(1);
    NetworkImpl net;

    IConnectableLayer *const layer = net.AddInputLayer(1);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckOutputLayerVisitorBindingIdAndName")
{
    const char* layerName = "OutputLayer";
    TestOutputLayerVisitor visitor(1, layerName);
    NetworkImpl net;

    IConnectableLayer *const layer = net.AddOutputLayer(1, layerName);
    layer->ExecuteStrategy(visitor);
}

TEST_CASE("CheckOutputLayerVisitorBindingIdAndNameNull")
{
    TestOutputLayerVisitor visitor(1);
    NetworkImpl net;

    IConnectableLayer *const layer = net.AddOutputLayer(1);
    layer->ExecuteStrategy(visitor);
}

}

} //namespace armnn