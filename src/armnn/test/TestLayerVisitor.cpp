//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TestLayerVisitor.hpp"

#include <doctest/doctest.h>

namespace armnn
{

void TestLayerVisitor::CheckLayerName(const char* name)
{
    if (name == nullptr)
    {
        CHECK(m_LayerName == nullptr);
    }
    else if (m_LayerName == nullptr)
    {
        CHECK(name == nullptr);
    }
    else
    {
        CHECK_EQ(std::string(m_LayerName), std::string(name));
    }
}

void TestLayerVisitor::CheckLayerPointer(const IConnectableLayer* layer)
{
    CHECK(layer != nullptr);
}

void TestLayerVisitor::CheckConstTensors(const ConstTensor& expected, const ConstTensor& actual)
{
    CHECK(expected.GetInfo() == actual.GetInfo());
    CHECK(expected.GetNumDimensions() == actual.GetNumDimensions());
    CHECK(expected.GetNumElements() == actual.GetNumElements());
    CHECK(expected.GetNumBytes() == actual.GetNumBytes());
    if (expected.GetNumBytes() == actual.GetNumBytes())
    {
        //check data is the same byte by byte
        const unsigned char* expectedPtr = static_cast<const unsigned char*>(expected.GetMemoryArea());
        const unsigned char* actualPtr = static_cast<const unsigned char*>(actual.GetMemoryArea());
        for (unsigned int i = 0; i < expected.GetNumBytes(); i++)
        {
            CHECK(*(expectedPtr + i) == *(actualPtr + i));
        }
    }
}

void TestLayerVisitor::CheckOptionalConstTensors(const Optional<ConstTensor>& expected,
                                                 const Optional<ConstTensor>& actual)
{
    CHECK(expected.has_value() == actual.has_value());
    if (expected.has_value() && actual.has_value())
    {
        CheckConstTensors(expected.value(), actual.value());
    }
}

} //namespace armnn
