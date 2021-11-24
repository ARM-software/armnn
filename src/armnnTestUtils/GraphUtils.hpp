//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <Graph.hpp>

#include <string>


bool GraphHasNamedLayer(const armnn::Graph& graph, const std::string& name);

armnn::Layer* GetFirstLayerWithName(armnn::Graph& graph, const std::string& name);

bool CheckNumberOfInputSlot(armnn::Layer* layer, unsigned int num);

bool CheckNumberOfOutputSlot(armnn::Layer* layer, unsigned int num);

bool IsConnected(armnn::Layer* srcLayer, armnn::Layer* destLayer,
                 unsigned int srcSlot, unsigned int destSlot,
                 const armnn::TensorInfo& expectedTensorInfo);

bool CheckOrder(const armnn::Graph& graph, const armnn::Layer* first, const armnn::Layer* second);

