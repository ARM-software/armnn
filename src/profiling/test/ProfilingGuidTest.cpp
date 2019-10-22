//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingGuid.hpp"

#include <boost/test/unit_test.hpp>

using namespace armnn::profiling;

BOOST_AUTO_TEST_SUITE(ProfilingGuidTests)

BOOST_AUTO_TEST_CASE(GuidTest)
{
    ProfilingGuid guid0(0);
    ProfilingGuid guid1(1);
    ProfilingGuid guid2(1);

    BOOST_TEST(guid0 != guid1);
    BOOST_TEST(guid1 == guid2);
    BOOST_TEST(guid0 < guid1);
    BOOST_TEST(guid0 <= guid1);
    BOOST_TEST(guid1 <= guid2);
    BOOST_TEST(guid1 > guid0);
    BOOST_TEST(guid1 >= guid0);
    BOOST_TEST(guid1 >= guid2);
}

BOOST_AUTO_TEST_CASE(StaticGuidTest)
{
    ProfilingStaticGuid guid0(0);
    ProfilingStaticGuid guid1(1);
    ProfilingStaticGuid guid2(1);

    BOOST_TEST(guid0 != guid1);
    BOOST_TEST(guid1 == guid2);
    BOOST_TEST(guid0 < guid1);
    BOOST_TEST(guid0 <= guid1);
    BOOST_TEST(guid1 <= guid2);
    BOOST_TEST(guid1 > guid0);
    BOOST_TEST(guid1 >= guid0);
    BOOST_TEST(guid1 >= guid2);
}

BOOST_AUTO_TEST_CASE(DynamicGuidTest)
{
    ProfilingDynamicGuid guid0(0);
    ProfilingDynamicGuid guid1(1);
    ProfilingDynamicGuid guid2(1);

    BOOST_TEST(guid0 != guid1);
    BOOST_TEST(guid1 == guid2);
    BOOST_TEST(guid0 < guid1);
    BOOST_TEST(guid0 <= guid1);
    BOOST_TEST(guid1 <= guid2);
    BOOST_TEST(guid1 > guid0);
    BOOST_TEST(guid1 >= guid0);
    BOOST_TEST(guid1 >= guid2);
}

BOOST_AUTO_TEST_SUITE_END()
