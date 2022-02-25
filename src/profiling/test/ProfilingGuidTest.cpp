//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Types.hpp>

#include <common/include/LabelsAndEventClasses.hpp>

#include <algorithm>
#include <functional>
#include <set>
#include <doctest/doctest.h>
#include <fmt/format.h>
#include <thread>

using namespace arm::pipe;

TEST_SUITE("ProfilingGuidTests")
{
TEST_CASE("GuidTest")
{
    ProfilingGuid guid0(0);
    ProfilingGuid guid1(1);
    ProfilingGuid guid2(1);

    CHECK(guid0 != guid1);
    CHECK(guid1 == guid2);
    CHECK(guid0 < guid1);
    CHECK(guid0 <= guid1);
    CHECK(guid1 <= guid2);
    CHECK(guid1 > guid0);
    CHECK(guid1 >= guid0);
    CHECK(guid1 >= guid2);
}

TEST_CASE("StaticGuidTest")
{
    ProfilingStaticGuid guid0(0);
    ProfilingStaticGuid guid1(1);
    ProfilingStaticGuid guid2(1);

    CHECK(guid0 != guid1);
    CHECK(guid1 == guid2);
    CHECK(guid0 < guid1);
    CHECK(guid0 <= guid1);
    CHECK(guid1 <= guid2);
    CHECK(guid1 > guid0);
    CHECK(guid1 >= guid0);
    CHECK(guid1 >= guid2);
}

TEST_CASE("DynamicGuidTest")
{
    ProfilingDynamicGuid guid0(0);
    ProfilingDynamicGuid guid1(1);
    ProfilingDynamicGuid guid2(1);

    CHECK(guid0 != guid1);
    CHECK(guid1 == guid2);
    CHECK(guid0 < guid1);
    CHECK(guid0 <= guid1);
    CHECK(guid1 <= guid2);
    CHECK(guid1 > guid0);
    CHECK(guid1 >= guid0);
    CHECK(guid1 >= guid2);
}

void CheckStaticGuid(uint64_t guid, uint64_t expectedGuid)
{
    CHECK(guid == expectedGuid);
    CHECK(guid >= MIN_STATIC_GUID);
}

void CheckDynamicGuid(uint64_t guid, uint64_t expectedGuid)
{
    CHECK(guid == expectedGuid);
    CHECK(guid < MIN_STATIC_GUID);
}

TEST_CASE("StaticGuidGeneratorCollisionTest")
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
            if (guid == ProfilingGuid(MIN_STATIC_GUID))
            {
                WARN("MIN_STATIC_GUID returned more than once from GenerateStaticId.");
            }
            else
            {
                FAIL(fmt::format("GUID collision occurred: {} -> {}", str, guid));
            }
            break;
        }
        guids.insert(guid);
    }
}

TEST_CASE("StaticGuidGeneratorTest")
{
    ProfilingGuidGenerator generator;

    ProfilingStaticGuid staticGuid0 = generator.GenerateStaticId("name");
    CheckStaticGuid(staticGuid0, LabelsAndEventClasses::NAME_GUID);
    CHECK(staticGuid0 != generator.GenerateStaticId("Name"));

    ProfilingStaticGuid staticGuid1 = generator.GenerateStaticId("type");
    CheckStaticGuid(staticGuid1, LabelsAndEventClasses::TYPE_GUID);
    CHECK(staticGuid1 != generator.GenerateStaticId("Type"));

    ProfilingStaticGuid staticGuid2 = generator.GenerateStaticId("index");
    CheckStaticGuid(staticGuid2, LabelsAndEventClasses::INDEX_GUID);
    CHECK(staticGuid2 != generator.GenerateStaticId("Index"));
}

TEST_CASE("DynamicGuidGeneratorTest")
{
    ProfilingGuidGenerator generator;

    for (unsigned int i = 0; i < 100; ++i)
    {
        ProfilingDynamicGuid guid = generator.NextGuid();
        CheckDynamicGuid(guid, i);
    }
}

void GenerateProfilingGUID(ProfilingGuidGenerator& guidGenerator)
{
    for (int i = 0; i < 1000; ++i)
    {
        guidGenerator.NextGuid();
    }
}

TEST_CASE("ProfilingGuidThreadTest")
{
    ProfilingGuidGenerator profilingGuidGenerator;
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < 3; ++i)
    {
        threads.push_back(std::thread(GenerateProfilingGUID, std::ref(profilingGuidGenerator)));
    }
    std::for_each(threads.begin(), threads.end(), [](std::thread& theThread)
    {
        theThread.join();
    });

    uint64_t guid = profilingGuidGenerator.NextGuid();
    CHECK(guid == 3000u);
}

}
