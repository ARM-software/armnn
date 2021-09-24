//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "TestLayerVisitor.hpp"

namespace
{

#define DECLARE_TEST_NAME_ONLY_LAYER_VISITOR_CLASS(name) \
class Test##name##LayerVisitor : public armnn::TestLayerVisitor \
{ \
public: \
    explicit Test##name##LayerVisitor(const char* layerName = nullptr) : armnn::TestLayerVisitor(layerName) {}; \
    \
    void ExecuteStrategy(const armnn::IConnectableLayer* layer, \
                         const armnn::BaseDescriptor& descriptor, \
                         const std::vector<armnn::ConstTensor>& constants, \
                         const char* layerName, \
                         const armnn::LayerBindingId id = 0) override \
    { \
        armnn::IgnoreUnused(descriptor, constants, id); \
        switch (layer->GetType()) \
        { \
            case armnn::LayerType::name: \
            { \
                CheckLayerPointer(layer); \
                CheckLayerName(layerName); \
                break; \
            } \
            default: \
            { \
                m_DefaultStrategy.Apply(GetLayerTypeAsCString(layer->GetType())); \
            } \
        } \
    } \
    \
};

} // anonymous namespace

DECLARE_TEST_NAME_ONLY_LAYER_VISITOR_CLASS(Addition)
DECLARE_TEST_NAME_ONLY_LAYER_VISITOR_CLASS(Dequantize)
DECLARE_TEST_NAME_ONLY_LAYER_VISITOR_CLASS(Division)
DECLARE_TEST_NAME_ONLY_LAYER_VISITOR_CLASS(Floor)
DECLARE_TEST_NAME_ONLY_LAYER_VISITOR_CLASS(Maximum)
DECLARE_TEST_NAME_ONLY_LAYER_VISITOR_CLASS(Merge)
DECLARE_TEST_NAME_ONLY_LAYER_VISITOR_CLASS(Minimum)
DECLARE_TEST_NAME_ONLY_LAYER_VISITOR_CLASS(Multiplication)
DECLARE_TEST_NAME_ONLY_LAYER_VISITOR_CLASS(Prelu)
DECLARE_TEST_NAME_ONLY_LAYER_VISITOR_CLASS(Quantize)
DECLARE_TEST_NAME_ONLY_LAYER_VISITOR_CLASS(Rank)
DECLARE_TEST_NAME_ONLY_LAYER_VISITOR_CLASS(Subtraction)
DECLARE_TEST_NAME_ONLY_LAYER_VISITOR_CLASS(Switch)
