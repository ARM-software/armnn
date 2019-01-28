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
};

} //namespace armnn