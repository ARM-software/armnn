// Copyright © 2026 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace armnn
{
struct Sme2CmeInfo
{
    bool m_CountKnown = false;
    uint32_t m_Count = 0;
};

// SMIDR affinity identifies shared SMCUs, so repeated reads from PEs must
// not multiply their count. NSMC encodes count minus one; 15 is reserved.
inline Sme2CmeInfo decode_sme2_cme_info(const std::vector<uint64_t>& values)
{
    if (values.empty()) { return {}; }
    uint32_t privateUnits = 0;
    std::map<uint32_t, uint32_t> sharedGroups;
    for (const uint64_t smidr : values)
    {
        const uint32_t nsmc = (smidr >> 56U) & 0xFU;
        const uint32_t sh = (smidr >> 13U) & 0x3U;
        const uint32_t affinity2 = (smidr >> 32U) & 0xFFFFFU;
        const uint32_t affinity = smidr & 0xFFFU;
        const uint32_t groupId = (affinity2 << 12U) | affinity;
        if (nsmc == 0xFU || sh == 0x1U) { return {}; }
        // SH=0 uses the low Affinity field alone to indicate sharing.
        if (sh == 0x2U || (sh == 0x0U && affinity == 0U))
        {
            if (nsmc != 0U) { return {}; }
            ++privateUnits;
            continue;
        }
        const auto inserted = sharedGroups.emplace(groupId, nsmc);
        if (!inserted.second && inserted.first->second != nsmc) { return {}; }
    }
    uint32_t count = privateUnits;
    for (const auto& group : sharedGroups) { count += group.second + 1U; }
    return {true, count};
}

inline std::vector<uint32_t> present_cpu_ids(const std::string& text)
{
    std::vector<uint32_t> cpus;
    size_t start = 0;
    auto number = [](const std::string& value) -> uint32_t
    {
        if (value.empty() || !std::all_of(value.begin(), value.end(), [](unsigned char c)
            { return std::isdigit(c); })) { throw std::invalid_argument("CPU number"); }
        const unsigned long parsed = std::stoul(value);
        if (parsed > 65535UL) { throw std::out_of_range("CPU number"); }
        return static_cast<uint32_t>(parsed);
    };
    try
    {
        while (start < text.size())
        {
            const size_t comma = text.find(',', start);
            const std::string token = text.substr(start, comma == std::string::npos ? comma : comma - start);
            const size_t dash = token.find('-');
            const uint32_t first = number(token.substr(0, dash));
            const uint32_t last = dash == std::string::npos ? first : number(token.substr(dash + 1));
            if (last < first) { return {}; }
            for (uint32_t cpu = first; cpu <= last; ++cpu) { cpus.push_back(cpu); }
            if (comma == std::string::npos) { break; }
            start = comma + 1;
            if (start == text.size()) { return {}; }
        }
    }
    catch (const std::exception&) { return {}; }
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    return cpus;
}

inline Sme2CmeInfo detect_num_smcus()
{
#if defined(__linux__) && defined(__aarch64__) && defined(HWCAP2_SME2)
    if (!(getauxval(AT_HWCAP2) & HWCAP2_SME2)) { return {}; }
    std::ifstream present("/sys/devices/system/cpu/present");
    std::string line;
    if (!std::getline(present, line)) { return {}; }
    const auto cpus = present_cpu_ids(line);
    if (cpus.empty()) { return {}; }
    std::vector<uint64_t> values;
    for (const uint32_t cpu : cpus)
    {
        // Include present but offline PEs; Linux caches these ID registers.
        std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                           "/regs/identification/smidr_el1");
        if (!std::getline(file, line)) { return {}; }
        try
        {
            size_t consumed = 0;
            const uint64_t smidr = std::stoull(line, &consumed, 16);
            while (consumed < line.size() && std::isspace(static_cast<unsigned char>(line[consumed])))
            { ++consumed; }
            if (consumed != line.size()) { return {}; }
            values.push_back(smidr);
        }
        catch (const std::exception&) { return {}; }
    }
    return decode_sme2_cme_info(values);
#else
    return {};
#endif
}
} // namespace armnn
