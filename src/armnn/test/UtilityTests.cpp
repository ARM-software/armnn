//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>

#define ARMNN_POLYMORPHIC_CAST_TESTABLE
#define ARMNN_NUMERIC_CAST_TESTABLE

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <armnn/Exceptions.hpp>

#include <limits>

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


BOOST_AUTO_TEST_CASE(PolymorphicPointerDowncast_SharedPointer)
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

    std::shared_ptr<Base> base1 = std::make_shared<Child1>();

    std::shared_ptr<Child1> ptr1 = std::static_pointer_cast<Child1>(base1);
    BOOST_CHECK(ptr1);
    BOOST_CHECK_NO_THROW(armnn::PolymorphicPointerDowncast<Child1>(base1));
    BOOST_CHECK(armnn::PolymorphicPointerDowncast<Child1>(base1) == ptr1);

    auto ptr2 = std::dynamic_pointer_cast<Child2>(base1);
    BOOST_CHECK(!ptr2);
    BOOST_CHECK_THROW(armnn::PolymorphicPointerDowncast<Child2>(base1), std::bad_cast);

    armnn::IgnoreUnused(ptr1, ptr2);
}


BOOST_AUTO_TEST_CASE(PolymorphicPointerDowncast_BuildInPointer)
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
    BOOST_CHECK_NO_THROW(armnn::PolymorphicPointerDowncast<Child1>(base1));
    BOOST_CHECK(armnn::PolymorphicPointerDowncast<Child1>(base1) == ptr1);

    auto ptr2 = dynamic_cast<Child2*>(base1);
    BOOST_CHECK(ptr2 == nullptr);
    BOOST_CHECK_THROW(armnn::PolymorphicPointerDowncast<Child2>(base1), std::bad_cast);

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

    BOOST_CHECK_NO_THROW(numeric_cast<int16_t>(1U << 8));
    BOOST_CHECK_NO_THROW(numeric_cast<int16_t>(1U << 14));

    // To 32 bit
    BOOST_CHECK_NO_THROW(numeric_cast<uint32_t>(1));
    BOOST_CHECK_NO_THROW(numeric_cast<uint32_t>(1 << 8));
    BOOST_CHECK_NO_THROW(numeric_cast<uint32_t>(1L << 16));
    BOOST_CHECK_NO_THROW(numeric_cast<uint32_t>(1LL << 31));

    BOOST_CHECK_NO_THROW(numeric_cast<int32_t>(-1));
    BOOST_CHECK_NO_THROW(numeric_cast<int32_t>((1L << 8)*-1));
    BOOST_CHECK_NO_THROW(numeric_cast<int32_t>((1L << 16)*-1));
    BOOST_CHECK_NO_THROW(numeric_cast<int32_t>((1LL << 31)*-1));

    BOOST_CHECK_NO_THROW(numeric_cast<int32_t>(1U));
    BOOST_CHECK_NO_THROW(numeric_cast<int32_t>(1U << 8));
    BOOST_CHECK_NO_THROW(numeric_cast<int32_t>(1U << 16));
    BOOST_CHECK_NO_THROW(numeric_cast<int32_t>(1U << 30));

    float float_max = std::numeric_limits<float>::max();
    float float_min = std::numeric_limits<float>::lowest();
    auto int8_max = std::numeric_limits<int8_t>::max();
    auto int16_max = std::numeric_limits<int16_t>::max();
    auto int32_max = std::numeric_limits<int32_t>::max();
    auto int8_min = std::numeric_limits<int8_t>::lowest();
    auto int16_min = std::numeric_limits<int16_t>::lowest();
    auto int32_min = std::numeric_limits<int32_t>::lowest();
    auto uint8_max = std::numeric_limits<uint8_t>::max();
    auto uint16_max = std::numeric_limits<uint16_t>::max();
    auto uint32_max = std::numeric_limits<uint32_t>::max();
    auto double_max = std::numeric_limits<double>::max();

    // Float to signed integer
    BOOST_CHECK_NO_THROW(numeric_cast<int32_t>(1.324f));
    BOOST_CHECK(1 == numeric_cast<int32_t>(1.324f));
    BOOST_CHECK_NO_THROW(numeric_cast<int32_t>(-1.0f));
    BOOST_CHECK(-1 == numeric_cast<int32_t>(-1.0f));

    BOOST_CHECK_NO_THROW(numeric_cast<int8_t>(static_cast<float>(int8_max)));
    BOOST_CHECK_NO_THROW(numeric_cast<int16_t>(static_cast<float>(int16_max)));
    BOOST_CHECK_NO_THROW(numeric_cast<int32_t>(static_cast<float>(int32_max)));

    BOOST_CHECK_THROW(numeric_cast<int8_t>(float_max), std::bad_cast);
    BOOST_CHECK_THROW(numeric_cast<int16_t>(float_max), std::bad_cast);
    BOOST_CHECK_THROW(numeric_cast<int32_t>(float_max), std::bad_cast);

    BOOST_CHECK_THROW(numeric_cast<int8_t>(float_min), std::bad_cast);
    BOOST_CHECK_THROW(numeric_cast<int16_t>(float_min), std::bad_cast);
    BOOST_CHECK_THROW(numeric_cast<int32_t>(float_min), std::bad_cast);

    // Signed integer to float
    BOOST_CHECK_NO_THROW(numeric_cast<float>(1));
    BOOST_CHECK(1.0 == numeric_cast<float>(1));
    BOOST_CHECK_NO_THROW(numeric_cast<float>(-1));
    BOOST_CHECK(-1.0 == numeric_cast<float>(-1));

    BOOST_CHECK_NO_THROW(numeric_cast<float>(int8_max));
    BOOST_CHECK_NO_THROW(numeric_cast<float>(int16_max));
    BOOST_CHECK_NO_THROW(numeric_cast<float>(int32_max));

    BOOST_CHECK_NO_THROW(numeric_cast<float>(int8_min));
    BOOST_CHECK_NO_THROW(numeric_cast<float>(int16_min));
    BOOST_CHECK_NO_THROW(numeric_cast<float>(int32_min));

    // Unsigned integer to float
    BOOST_CHECK_NO_THROW(numeric_cast<float>(1U));
    BOOST_CHECK(1.0 == numeric_cast<float>(1U));

    BOOST_CHECK_NO_THROW(numeric_cast<float>(uint8_max));
    BOOST_CHECK_NO_THROW(numeric_cast<float>(uint16_max));
    BOOST_CHECK_NO_THROW(numeric_cast<float>(uint32_max));

    // Float to unsigned integer
    BOOST_CHECK_NO_THROW(numeric_cast<uint32_t>(1.43243f));
    BOOST_CHECK(1 == numeric_cast<uint32_t>(1.43243f));

    BOOST_CHECK_THROW(numeric_cast<uint32_t>(-1.1f), std::bad_cast);
    BOOST_CHECK_THROW(numeric_cast<uint32_t>(float_max), std::bad_cast);

    // Double checks
    BOOST_CHECK_THROW(numeric_cast<int32_t>(double_max), std::bad_cast);
    BOOST_CHECK_THROW(numeric_cast<int32_t>(double_max), std::bad_cast);
    BOOST_CHECK_THROW(numeric_cast<float>(double_max), std::bad_cast);
    BOOST_CHECK_NO_THROW(numeric_cast<double>(int32_max));
    BOOST_CHECK_NO_THROW(numeric_cast<long double>(int32_max));

    }

BOOST_AUTO_TEST_SUITE_END()
