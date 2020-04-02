//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include <boost/core/lightweight_test.hpp>
#include <boost/polymorphic_cast.hpp>

#define ARMNN_POLYMORPHIC_CAST_TESTABLE
#define ARMNN_NUMERIC_CAST_TESTABLE

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/utility/NumericCast.hpp>

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
    BOOST_CHECK_NO_THROW(armnn::PolymorphicDowncast<Child1*>(base1));
    BOOST_CHECK(armnn::PolymorphicDowncast<Child1*>(base1) == ptr1);

    auto ptr2 = dynamic_cast<Child2*>(base1);
    BOOST_CHECK(ptr2 == nullptr);
    BOOST_CHECK_THROW(armnn::PolymorphicDowncast<Child2*>(base1), std::bad_cast);

    armnn::IgnoreUnused(ptr1, ptr2);
}


BOOST_AUTO_TEST_CASE(NumericCast)
{
    using namespace armnn;

    // To 8 bit
    BOOST_CHECK_THROW(numeric_cast<unsigned char>(-1), std::bad_cast);
    BOOST_CHECK_THROW(numeric_cast<unsigned char>(1 << 8), std::bad_cast);
    BOOST_CHECK_THROW(numeric_cast<unsigned char>(1L << 16), std::bad_cast);
    BOOST_CHECK_THROW(numeric_cast<unsigned char>(1LL << 32), std::bad_cast);

    BOOST_CHECK_THROW(numeric_cast<signed char>((1L << 8)*-1), std::bad_cast);
    BOOST_CHECK_THROW(numeric_cast<signed char>((1L << 15)*-1), std::bad_cast);
    BOOST_CHECK_THROW(numeric_cast<signed char>((1LL << 31)*-1), std::bad_cast);

    BOOST_CHECK_NO_THROW(numeric_cast<unsigned char>(1U));
    BOOST_CHECK_NO_THROW(numeric_cast<unsigned char>(1L));
    BOOST_CHECK_NO_THROW(numeric_cast<signed char>(-1));
    BOOST_CHECK_NO_THROW(numeric_cast<signed char>(-1L));
    BOOST_CHECK_NO_THROW(numeric_cast<signed char>((1 << 7)*-1));

    // To 16 bit
    BOOST_CHECK_THROW(numeric_cast<uint16_t>(-1), std::bad_cast);
    BOOST_CHECK_THROW(numeric_cast<uint16_t>(1L << 16), std::bad_cast);
    BOOST_CHECK_THROW(numeric_cast<uint16_t>(1LL << 32), std::bad_cast);

    BOOST_CHECK_THROW(numeric_cast<int16_t>(1L << 15), std::bad_cast);
    BOOST_CHECK_THROW(numeric_cast<int16_t>(1LL << 31), std::bad_cast);

    BOOST_CHECK_NO_THROW(numeric_cast<uint16_t>(1L << 8));
    BOOST_CHECK_NO_THROW(numeric_cast<int16_t>(1L << 7));
    BOOST_CHECK_NO_THROW(numeric_cast<int16_t>((1L << 15)*-1));

    // To 32 bit
    BOOST_CHECK_NO_THROW(numeric_cast<uint32_t>(1));
    BOOST_CHECK_NO_THROW(numeric_cast<uint32_t>(1 << 8));
    BOOST_CHECK_NO_THROW(numeric_cast<uint32_t>(1L << 16));
    BOOST_CHECK_NO_THROW(numeric_cast<uint32_t>(1LL << 31));

    BOOST_CHECK_NO_THROW(numeric_cast<int32_t>(-1));
    BOOST_CHECK_NO_THROW(numeric_cast<int32_t>((1L << 8)*-1));
    BOOST_CHECK_NO_THROW(numeric_cast<int32_t>((1L << 16)*-1));
    BOOST_CHECK_NO_THROW(numeric_cast<int32_t>((1LL << 31)*-1));
}

BOOST_AUTO_TEST_SUITE_END()
