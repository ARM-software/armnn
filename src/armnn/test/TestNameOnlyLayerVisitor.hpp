//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "TestLayerVisitor.hpp"

namespace
{

// Defines a visitor function with 1 required parameter to be used
// with layers that do not have a descriptor
#define VISIT_METHOD_1_PARAM(name) \
void Visit##name##Layer(const armnn::IConnectableLayer* layer, const char* layerName = nullptr) override

// Defines a visitor function with 2 required parameters to be used
// with layers that have a descriptor
#define VISIT_METHOD_2_PARAM(name) \
void Visit##name##Layer(const armnn::IConnectableLayer* layer, \
                        const armnn::name##Descriptor&, \
                        const char* layerName = nullptr) override

#define TEST_LAYER_VISITOR(name, numVisitorParams) \
class Test##name##LayerVisitor : public armnn::TestLayerVisitor \
{ \
public: \
    explicit Test##name##LayerVisitor(const char* layerName = nullptr) : armnn::TestLayerVisitor(layerName) {}; \
    \
    VISIT_METHOD_##numVisitorParams##_PARAM(name) \
    { \
        CheckLayerPointer(layer); \
        CheckLayerName(layerName); \
    } \
};

// Defines a test layer visitor class for a layer, of a given name,
// that does not require a descriptor
#define TEST_LAYER_VISITOR_1_PARAM(name) TEST_LAYER_VISITOR(name, 1)

// Defines a test layer visitor class for a layer, of a given name,
// that requires a descriptor
#define TEST_LAYER_VISITOR_2_PARAM(name) TEST_LAYER_VISITOR(name, 2)

} // anonymous namespace

TEST_LAYER_VISITOR_1_PARAM(Addition)
TEST_LAYER_VISITOR_1_PARAM(Division)
TEST_LAYER_VISITOR_1_PARAM(Equal)
TEST_LAYER_VISITOR_1_PARAM(Floor)
TEST_LAYER_VISITOR_1_PARAM(Gather)
TEST_LAYER_VISITOR_1_PARAM(Greater)
TEST_LAYER_VISITOR_1_PARAM(Maximum)
TEST_LAYER_VISITOR_1_PARAM(Minimum)
TEST_LAYER_VISITOR_1_PARAM(Multiplication)
TEST_LAYER_VISITOR_1_PARAM(Rsqrt)
TEST_LAYER_VISITOR_2_PARAM(Slice)
TEST_LAYER_VISITOR_1_PARAM(Subtraction)
