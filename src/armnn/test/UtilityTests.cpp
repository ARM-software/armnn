//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include <boost/core/lightweight_test.hpp>
#include <boost/polymorphic_cast.hpp>

#define ARMNN_POLYMORPHIC_CAST_TESTABLE

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <armnn/Exceptions.hpp>

// Tests of include/Utility files
BOOST_AUTO_TEST_SUITE(UtilityTests)

BOOST_AUTO_TEST_CASE(PolymorphicDowncast)
{
    using namespace armnn;
    class Base
    {
    public:
        virtual ~Base(){}
        float v;
    };

    class Child1 : public Base
    {
    public:
        int j;
    };

    class Child2 : public Base
    {
    public:
        char b;
    };

    Child1 child1;
    Base* base1 = &child1;
    auto ptr1 = dynamic_cast<Child1*>(base1);
    BOOST_CHECK(ptr1 != nullptr);
    BOOST_CHECK_NO_THROW(polymorphic_downcast<Child1*>(base1));
    BOOST_CHECK(polymorphic_downcast<Child1*>(base1) == ptr1);

    auto ptr2 = dynamic_cast<Child2*>(base1);
    BOOST_CHECK(ptr2 == nullptr);
    BOOST_CHECK_THROW(polymorphic_downcast<Child2*>(base1), std::bad_cast);

    armnn::IgnoreUnused(ptr1, ptr2);
}

BOOST_AUTO_TEST_SUITE_END()
