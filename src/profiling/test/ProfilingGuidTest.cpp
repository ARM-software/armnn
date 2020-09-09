//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Types.hpp>

#include "LabelsAndEventClasses.hpp"
#include "ProfilingGuidGenerator.hpp"

#include <set>

#include <boost/test/unit_test.hpp>
#include <fmt/format.h>
#include <thread>

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
            // If we're running on a 32bit system it is more likely to get a GUID clash over 1 million executions.
            // We can generally detect this when the GUID turns out to be MIN_STATIC_GUID. Output a warning
            // message rather than error in this case.
            if (guid == ProfilingGuid(armnn::profiling::MIN_STATIC_GUID))
            {
                BOOST_WARN("MIN_STATIC_GUID returned more than once from GenerateStaticId.");
            } 
            else
            {
                BOOST_ERROR(fmt::format("GUID collision occurred: {} -> {}", str, guid));
            }
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

BOOST_AUTO_TEST_CASE (ProfilingGuidThreadTest)
{
    ProfilingGuidGenerator profilingGuidGenerator;

    auto guidGenerator = [&profilingGuidGenerator]()
    {
        for (int i = 0; i < 1000; ++i)
        {
            profilingGuidGenerator.NextGuid();
        }
    };

    std::thread t1(guidGenerator);
    std::thread t2(guidGenerator);
    std::thread t3(guidGenerator);

    t1.join();
    t2.join();
    t3.join();

    uint64_t guid = profilingGuidGenerator.NextGuid();
    BOOST_CHECK(guid == 3000u);
}

BOOST_AUTO_TEST_SUITE_END()
