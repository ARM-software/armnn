//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Types.hpp>

#include "LabelsAndEventClasses.hpp"
#include "ProfilingGuidGenerator.hpp"

#include <set>

#include <boost/test/unit_test.hpp>
#include <boost/format.hpp>

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

void CheckStaticGuid(uint64_t guid, uint64_t expectedGuid)
{
    BOOST_TEST(guid == expectedGuid);
    BOOST_TEST(guid >= MIN_STATIC_GUID);
}

void CheckDynamicGuid(uint64_t guid, uint64_t expectedGuid)
{
    BOOST_TEST(guid == expectedGuid);
    BOOST_TEST(guid < MIN_STATIC_GUID);
}

BOOST_AUTO_TEST_CASE(StaticGuidGeneratorCollisionTest)
{
    ProfilingGuidGenerator generator;
    std::set<uint64_t> guids;
    for ( int i = 0; i < 100000; ++i )
    {
        std::stringstream ss;
        ss << i;
        std::string str = ss.str();
        ProfilingStaticGuid guid = generator.GenerateStaticId(str.c_str());
        if (guids.find(guid) != guids.end())
        {
            BOOST_ERROR(boost::str(boost::format("GUID collision occurred: %1% -> %2%") % str % guid));
            break;
        }
        guids.insert(guid);
    }
}

BOOST_AUTO_TEST_CASE(StaticGuidGeneratorTest)
{
    ProfilingGuidGenerator generator;

    ProfilingStaticGuid staticGuid0 = generator.GenerateStaticId("name");
    CheckStaticGuid(staticGuid0, LabelsAndEventClasses::NAME_GUID);
    BOOST_TEST(staticGuid0 != generator.GenerateStaticId("Name"));

    ProfilingStaticGuid staticGuid1 = generator.GenerateStaticId("type");
    CheckStaticGuid(staticGuid1, LabelsAndEventClasses::TYPE_GUID);
    BOOST_TEST(staticGuid1 != generator.GenerateStaticId("Type"));

    ProfilingStaticGuid staticGuid2 = generator.GenerateStaticId("index");
    CheckStaticGuid(staticGuid2, LabelsAndEventClasses::INDEX_GUID);
    BOOST_TEST(staticGuid2 != generator.GenerateStaticId("Index"));
}

BOOST_AUTO_TEST_CASE(DynamicGuidGeneratorTest)
{
    ProfilingGuidGenerator generator;

    for (unsigned int i = 0; i < 100; ++i)
    {
        ProfilingDynamicGuid guid = generator.NextGuid();
        CheckDynamicGuid(guid, i);
    }
}

BOOST_AUTO_TEST_SUITE_END()
