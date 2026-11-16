// Copyright © 2026 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT

#include "../Sme2CmeInfo.hpp"

#include <doctest/doctest.h>

namespace
{

constexpr uint64_t TwoSharedSmcus = 0x0100000041126000ULL;
constexpr uint64_t OneSharedSmcu = 0x0000000041116000ULL;

void CheckSmcuCount(const std::vector<uint64_t>& values, uint32_t expected)
{
    const auto info = armnn::decode_sme2_cme_info(values);
    CHECK(info.m_CountKnown);
    CHECK(info.m_Count == expected);
}

} // namespace

TEST_SUITE("Sme2CmeInfoTests")
{

TEST_CASE("DeduplicateSharedSmcus")
{
    CheckSmcuCount(std::vector<uint64_t>(10, TwoSharedSmcus), 2);
    CheckSmcuCount(std::vector<uint64_t>(10, OneSharedSmcu), 1);
}

TEST_CASE("SumDistinctSharedGroupsAndPrivateSmcus")
{
    CheckSmcuCount({TwoSharedSmcus, TwoSharedSmcus | 1U}, 4);
    CheckSmcuCount({TwoSharedSmcus, OneSharedSmcu | 1U}, 3);
    CheckSmcuCount({TwoSharedSmcus, 0x4000U, 0x4000U}, 4);
}

TEST_CASE("PrivateSmcusAreCountedPerCpu")
{
    CheckSmcuCount({0x4000U, 0x4000U}, 2);
    CheckSmcuCount({0U, 0U}, 2);
    // SH=00 and Affinity=0 means private even with nonzero Affinity2.
    CheckSmcuCount({0x0000000100000000ULL, 0x0000000100000000ULL}, 2);
}

TEST_CASE("LegacySharedSmcusUseTheFullAffinityGroupId")
{
    // SH=00 and nonzero Affinity means shared; Affinity2 distinguishes groups.
    constexpr uint64_t firstGroup = 0x0100000100000001ULL;
    constexpr uint64_t secondGroup = 0x0100000200000001ULL;
    CheckSmcuCount({firstGroup, firstGroup}, 2);
    CheckSmcuCount({firstGroup, secondGroup}, 4);
}

TEST_CASE("RejectUnknownOrInconsistentTopology")
{
    const std::vector<std::vector<uint64_t>> invalidValues =
    {
        {},
        {TwoSharedSmcus, OneSharedSmcu},
        {0x0f00000000006000ULL}, // Reserved NSMC.
        {0x2000U},             // Reserved SH.
        {0x0100000000004000ULL}, // Private unit with nonzero NSMC.
        {0x0100000100000000ULL}  // Legacy private unit with nonzero NSMC.
    };
    for (const auto& values : invalidValues)
    {
        const auto info = armnn::decode_sme2_cme_info(values);
        CHECK_FALSE(info.m_CountKnown);
        CHECK(info.m_Count == 0);
    }
}

TEST_CASE("ParsePresentCpuRanges")
{
    CHECK(armnn::present_cpu_ids("0-3,8,9") == std::vector<uint32_t>({0, 1, 2, 3, 8, 9}));
    CHECK(armnn::present_cpu_ids("8,0-2,1,8") == std::vector<uint32_t>({0, 1, 2, 8}));
    for (const auto* invalid : {"", "3-0", "0,", "0,,1", "-1", "0-2-3", "65536", "0 junk"})
    {
        CHECK(armnn::present_cpu_ids(invalid).empty());
    }
}

} // TEST_SUITE
