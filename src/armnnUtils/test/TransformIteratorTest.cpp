//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/utility/TransformIterator.hpp>

#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace armnn;

BOOST_AUTO_TEST_SUITE(TransformIteratorSuite)

namespace
{

static int square(const int val)
{
    return val * val;
}

static std::string concat(const std::string val)
{
    return val + "a";
}

BOOST_AUTO_TEST_CASE(TransformIteratorTest)
{
    struct WrapperTestClass
    {
        TransformIterator<decltype(&square), std::vector<int>::const_iterator> begin() const
        {
            return { m_Vec.begin(), &square };
        }

        TransformIterator<decltype(&square), std::vector<int>::const_iterator>  end() const
        {
            return { m_Vec.end(), &square };
        }

        const std::vector<int> m_Vec{1, 2, 3, 4, 5};
    };

    struct WrapperStringClass
    {
        TransformIterator<decltype(&concat), std::vector<std::string>::const_iterator> begin() const
        {
            return { m_Vec.begin(), &concat };
        }

        TransformIterator<decltype(&concat), std::vector<std::string>::const_iterator>  end() const
        {
            return { m_Vec.end(), &concat };
        }

        const std::vector<std::string> m_Vec{"a", "b", "c"};
    };

    WrapperStringClass wrapperStringClass;
    WrapperTestClass wrapperTestClass;
    int i = 1;

    for(auto val : wrapperStringClass)
    {
        BOOST_CHECK(val != "e");
        i++;
    }

    i = 1;
    for(auto val : wrapperTestClass)
    {
        BOOST_CHECK(val == square(i));
        i++;
    }

    i = 1;
    // Check original vector is unchanged
    for(auto val : wrapperTestClass.m_Vec)
    {
        BOOST_CHECK(val == i);
        i++;
    }

    std::vector<int> originalVec{1, 2, 3, 4, 5};

    auto transformBegin = MakeTransformIterator(originalVec.begin(), &square);
    auto transformEnd = MakeTransformIterator(originalVec.end(), &square);

    std::vector<int> transformedVec(transformBegin, transformEnd);

    i = 1;
    for(auto val : transformedVec)
    {
        BOOST_CHECK(val == square(i));
        i++;
    }
}

}

BOOST_AUTO_TEST_SUITE_END()
