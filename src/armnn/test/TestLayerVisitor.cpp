//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "TestLayerVisitor.hpp"

namespace armnn
{

void TestLayerVisitor::CheckLayerName(const char* name)
{
    if (name == nullptr)
    {
        BOOST_CHECK(m_LayerName == nullptr);
    }
    else if (m_LayerName == nullptr)
    {
        BOOST_CHECK(name == nullptr);
    }
    else
    {
        BOOST_CHECK_EQUAL(m_LayerName, name);
    }
}

void TestLayerVisitor::CheckLayerPointer(const IConnectableLayer* layer)
{
    BOOST_CHECK(layer != nullptr);
}

void TestLayerVisitor::CheckConstTensors(const ConstTensor& expected, const ConstTensor& actual)
{
    BOOST_CHECK(expected.GetInfo() == actual.GetInfo());
    BOOST_CHECK(expected.GetNumDimensions() == actual.GetNumDimensions());
    BOOST_CHECK(expected.GetNumElements() == actual.GetNumElements());
    BOOST_CHECK(expected.GetNumBytes() == actual.GetNumBytes());
    if (expected.GetNumBytes() == actual.GetNumBytes())
    {
        //check data is the same byte by byte
        const unsigned char* expectedPtr = static_cast<const unsigned char*>(expected.GetMemoryArea());
        const unsigned char* actualPtr = static_cast<const unsigned char*>(actual.GetMemoryArea());
        for (unsigned int i = 0; i < expected.GetNumBytes(); i++)
        {
            BOOST_CHECK(*(expectedPtr + i) == *(actualPtr + i));
        }
    }
}

void TestLayerVisitor::CheckOptionalConstTensors(const Optional<ConstTensor>& expected,
                                                 const Optional<ConstTensor>& actual)
{
    BOOST_CHECK(expected.has_value() == actual.has_value());
    if (expected.has_value() && actual.has_value())
    {
        CheckConstTensors(expected.value(), actual.value());
    }
}

} //namespace armnn
