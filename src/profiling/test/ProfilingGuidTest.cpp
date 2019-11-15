//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Types.hpp>

#include "LabelsAndEventClasses.hpp"
#include "ProfilingGuidGenerator.hpp"

#include <set>

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

std::string GenerateRandomString()
{
    // Random a string lengh from 3 - 100
    int minLength = 3;
    int maxLength = 100;

    // Random a character from lower case alphabets, upper case alphabets, numbers and special characters
    int minAscii = 32; // space 32
    int maxAscii = 126; // ~

    int stringLen = rand() % (maxLength - minLength + 1) + minLength;
    char str[stringLen + 1];
    for (int i = 0; i < stringLen; ++i)
    {
        int randAscii = rand() % (maxAscii - minAscii + 1) + minAscii;
        str[i] = char(randAscii);
    }
    str[stringLen] = '\0';
    return std::string(str);
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
    std::set<std::string> strs;
    std::map<uint64_t,std::string> guidMap;
    int collision = 0;
    for (int i = 0; i < 1000000; ++i)
    {
        std::string str = GenerateRandomString();
        if(strs.find(str) != strs.end())
        {
            continue;
        }
        strs.insert(str);
        ProfilingStaticGuid guid = generator.GenerateStaticId(str.c_str());
        if (guids.find(guid) != guids.end())
        {
            collision++;
        }
        guids.insert(guid);
    }
    BOOST_TEST(collision == 0);
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
