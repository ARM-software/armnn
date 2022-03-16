//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <common/include/Assert.hpp>
#include <common/include/CommonProfilingUtils.hpp>
#include <common/include/ProfilingException.hpp>

#include <iostream>
#include <limits>
#include <sstream>

namespace arm
{

namespace pipe
{
void ReadBytes(const unsigned char* buffer, unsigned int offset, unsigned int valueSize, uint8_t outValue[])
{
    ARM_PIPE_ASSERT(buffer);
    ARM_PIPE_ASSERT(outValue);

    for (unsigned int i = 0; i < valueSize; i++, offset++)
    {
        outValue[i] = static_cast<uint8_t>(buffer[offset]);
    }
}

uint64_t ReadUint64(const unsigned char* buffer, unsigned int offset)
{
    ARM_PIPE_ASSERT(buffer);

    uint64_t value = 0;
    value  = static_cast<uint64_t>(buffer[offset]);
    value |= static_cast<uint64_t>(buffer[offset + 1]) << 8;
    value |= static_cast<uint64_t>(buffer[offset + 2]) << 16;
    value |= static_cast<uint64_t>(buffer[offset + 3]) << 24;
    value |= static_cast<uint64_t>(buffer[offset + 4]) << 32;
    value |= static_cast<uint64_t>(buffer[offset + 5]) << 40;
    value |= static_cast<uint64_t>(buffer[offset + 6]) << 48;
    value |= static_cast<uint64_t>(buffer[offset + 7]) << 56;

    return value;
}

uint32_t ReadUint32(const unsigned char* buffer, unsigned int offset)
{
    ARM_PIPE_ASSERT(buffer);

    uint32_t value = 0;
    value  = static_cast<uint32_t>(buffer[offset]);
    value |= static_cast<uint32_t>(buffer[offset + 1]) << 8;
    value |= static_cast<uint32_t>(buffer[offset + 2]) << 16;
    value |= static_cast<uint32_t>(buffer[offset + 3]) << 24;
    return value;
}

uint16_t ReadUint16(const unsigned char* buffer, unsigned int offset)
{
    ARM_PIPE_ASSERT(buffer);

    uint32_t value = 0;
    value  = static_cast<uint32_t>(buffer[offset]);
    value |= static_cast<uint32_t>(buffer[offset + 1]) << 8;
    return static_cast<uint16_t>(value);
}

uint8_t ReadUint8(const unsigned char* buffer, unsigned int offset)
{
    ARM_PIPE_ASSERT(buffer);

    return buffer[offset];
}

void WriteBytes(unsigned char* buffer, unsigned int offset, const void* value, unsigned int valueSize)
{
    ARM_PIPE_ASSERT(buffer);
    ARM_PIPE_ASSERT(value);

    for (unsigned int i = 0; i < valueSize; i++, offset++)
    {
        buffer[offset] = *(reinterpret_cast<const unsigned char*>(value) + i);
    }
}

void WriteUint64(unsigned char* buffer, unsigned int offset, uint64_t value)
{
    ARM_PIPE_ASSERT(buffer);

    buffer[offset]     = static_cast<unsigned char>(value & 0xFF);
    buffer[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);
    buffer[offset + 2] = static_cast<unsigned char>((value >> 16) & 0xFF);
    buffer[offset + 3] = static_cast<unsigned char>((value >> 24) & 0xFF);
    buffer[offset + 4] = static_cast<unsigned char>((value >> 32) & 0xFF);
    buffer[offset + 5] = static_cast<unsigned char>((value >> 40) & 0xFF);
    buffer[offset + 6] = static_cast<unsigned char>((value >> 48) & 0xFF);
    buffer[offset + 7] = static_cast<unsigned char>((value >> 56) & 0xFF);
}

void WriteUint32(unsigned char* buffer, unsigned int offset, uint32_t value)
{
    ARM_PIPE_ASSERT(buffer);

    buffer[offset]     = static_cast<unsigned char>(value & 0xFF);
    buffer[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);
    buffer[offset + 2] = static_cast<unsigned char>((value >> 16) & 0xFF);
    buffer[offset + 3] = static_cast<unsigned char>((value >> 24) & 0xFF);
}

void WriteUint16(unsigned char* buffer, unsigned int offset, uint16_t value)
{
    ARM_PIPE_ASSERT(buffer);

    buffer[offset]     = static_cast<unsigned char>(value & 0xFF);
    buffer[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);
}

void WriteUint8(unsigned char* buffer, unsigned int offset, uint8_t value)
{
    ARM_PIPE_ASSERT(buffer);

    buffer[offset] = static_cast<unsigned char>(value);
}

std::string CentreAlignFormatting(const std::string& stringToPass, const int spacingWidth)
{
    std::stringstream outputStream, centrePadding;
    int padding = spacingWidth - static_cast<int>(stringToPass.size());

    for (int i = 0; i < padding / 2; ++i)
    {
        centrePadding << " ";
    }

    outputStream << centrePadding.str() << stringToPass << centrePadding.str();

    if (padding > 0 && padding %2 != 0)
    {
        outputStream << " ";
    }

    return outputStream.str();
}

void PrintDeviceDetails(const std::pair<const unsigned short, std::unique_ptr<Device>>& devicePair)
{
    std::string body;

    body.append(CentreAlignFormatting(devicePair.second->m_Name, 20));
    body.append(" | ");
    body.append(CentreAlignFormatting(std::to_string(devicePair.first), 13));
    body.append(" | ");
    body.append(CentreAlignFormatting(std::to_string(devicePair.second->m_Cores), 10));
    body.append("\n");

    std::cout << std::string(body.size(), '-') << "\n";
    std::cout<< body;
}

void PrintCounterSetDetails(const std::pair<const unsigned short, std::unique_ptr<CounterSet>>& counterSetPair)
{
    std::string body;

    body.append(CentreAlignFormatting(counterSetPair.second->m_Name, 20));
    body.append(" | ");
    body.append(CentreAlignFormatting(std::to_string(counterSetPair.first), 13));
    body.append(" | ");
    body.append(CentreAlignFormatting(std::to_string(counterSetPair.second->m_Count), 10));
    body.append("\n");

    std::cout << std::string(body.size(), '-') << "\n";

    std::cout<< body;
}

void PrintCounterDetails(std::shared_ptr<Counter>& counter)
{
    std::string body;

    body.append(CentreAlignFormatting(counter->m_Name, 20));
    body.append(" | ");
    body.append(CentreAlignFormatting(counter->m_Description, 50));
    body.append(" | ");
    body.append(CentreAlignFormatting(counter->m_Units, 14));
    body.append(" | ");
    body.append(CentreAlignFormatting(std::to_string(counter->m_Uid), 6));
    body.append(" | ");
    body.append(CentreAlignFormatting(std::to_string(counter->m_MaxCounterUid), 10));
    body.append(" | ");
    body.append(CentreAlignFormatting(std::to_string(counter->m_Class), 8));
    body.append(" | ");
    body.append(CentreAlignFormatting(std::to_string(counter->m_Interpolation), 14));
    body.append(" | ");
    body.append(CentreAlignFormatting(std::to_string(counter->m_Multiplier), 20));
    body.append(" | ");
    body.append(CentreAlignFormatting(std::to_string(counter->m_CounterSetUid), 16));
    body.append(" | ");
    body.append(CentreAlignFormatting(std::to_string(counter->m_DeviceUid), 14));

    body.append("\n");

    std::cout << std::string(body.size(), '-') << "\n";

    std::cout << body;
}

void PrintCategoryDetails(const std::unique_ptr<Category>& category,
                          std::unordered_map<unsigned short, std::shared_ptr<Counter>> counterMap)
{
    std::string categoryBody;
    std::string categoryHeader;

    categoryHeader.append(CentreAlignFormatting("Name", 20));
    categoryHeader.append(" | ");
    categoryHeader.append(CentreAlignFormatting("Event Count", 14));
    categoryHeader.append("\n");

    categoryBody.append(CentreAlignFormatting(category->m_Name, 20));
    categoryBody.append(" | ");
    categoryBody.append(CentreAlignFormatting(std::to_string(category->m_Counters.size()), 14));

    std::cout << "\n" << "\n";
    std::cout << CentreAlignFormatting("CATEGORY", static_cast<int>(categoryHeader.size()));
    std::cout << "\n";
    std::cout << std::string(categoryHeader.size(), '=') << "\n";

    std::cout << categoryHeader;

    std::cout << std::string(categoryBody.size(), '-') << "\n";

    std::cout << categoryBody;

    std::string counterHeader;

    counterHeader.append(CentreAlignFormatting("Counter Name", 20));
    counterHeader.append(" | ");
    counterHeader.append(CentreAlignFormatting("Description", 50));
    counterHeader.append(" | ");
    counterHeader.append(CentreAlignFormatting("Units", 14));
    counterHeader.append(" | ");
    counterHeader.append(CentreAlignFormatting("UID", 6));
    counterHeader.append(" | ");
    counterHeader.append(CentreAlignFormatting("Max UID", 10));
    counterHeader.append(" | ");
    counterHeader.append(CentreAlignFormatting("Class", 8));
    counterHeader.append(" | ");
    counterHeader.append(CentreAlignFormatting("Interpolation", 14));
    counterHeader.append(" | ");
    counterHeader.append(CentreAlignFormatting("Multiplier", 20));
    counterHeader.append(" | ");
    counterHeader.append(CentreAlignFormatting("Counter set UID", 16));
    counterHeader.append(" | ");
    counterHeader.append(CentreAlignFormatting("Device UID", 14));
    counterHeader.append("\n");

    std::cout << "\n" << "\n";
    std::cout << CentreAlignFormatting("EVENTS IN CATEGORY: " + category->m_Name,
                                       static_cast<int>(counterHeader.size()));
    std::cout << "\n";
    std::cout << std::string(counterHeader.size(), '=') << "\n";
    std::cout << counterHeader;
    for (auto& it: category->m_Counters) {
        auto search = counterMap.find(it);
        if(search != counterMap.end()) {
            PrintCounterDetails(search->second);
        }
    }
}

void PrintCounterDirectory(ICounterDirectory& counterDirectory)
{
    std::string devicesHeader;

    devicesHeader.append(CentreAlignFormatting("Device name", 20));
    devicesHeader.append(" | ");
    devicesHeader.append(CentreAlignFormatting("UID", 13));
    devicesHeader.append(" | ");
    devicesHeader.append(CentreAlignFormatting("Cores", 10));
    devicesHeader.append("\n");

    std::cout << "\n" << "\n";
    std::cout << CentreAlignFormatting("DEVICES", static_cast<int>(devicesHeader.size()));
    std::cout << "\n";
    std::cout << std::string(devicesHeader.size(), '=') << "\n";
    std::cout << devicesHeader;
    for (auto& it: counterDirectory.GetDevices()) {
        PrintDeviceDetails(it);
    }

    std::string counterSetHeader;

    counterSetHeader.append(CentreAlignFormatting("Counter set name", 20));
    counterSetHeader.append(" | ");
    counterSetHeader.append(CentreAlignFormatting("UID", 13));
    counterSetHeader.append(" | ");
    counterSetHeader.append(CentreAlignFormatting("Count", 10));
    counterSetHeader.append("\n");

    std::cout << "\n" << "\n";
    std::cout << CentreAlignFormatting("COUNTER SETS", static_cast<int>(counterSetHeader.size()));
    std::cout << "\n";
    std::cout << std::string(counterSetHeader.size(), '=') << "\n";

    std::cout << counterSetHeader;

    for (auto& it: counterDirectory.GetCounterSets()) {
        PrintCounterSetDetails(it);
    }

    auto counters = counterDirectory.GetCounters();
    for (auto& it: counterDirectory.GetCategories()) {
        PrintCategoryDetails(it, counters);
    }
    std::cout << "\n";
}

namespace
{

void ThrowIfCantGenerateNextUid(uint16_t uid, uint16_t cores = 0)
{
    // Check that it is possible to generate the next UID without causing an overflow
    switch (cores)
    {
    case 0:
    case 1:
        // Number of cores not specified or set to 1 (a value of zero indicates the device is not capable of
        // running multiple parallel workloads and will not provide multiple streams of data for each event)
        if (uid == std::numeric_limits<uint16_t>::max())
        {
            throw arm::pipe::ProfilingException("Generating the next UID for profiling would result in an overflow");
        }
        break;
    default: // cores > 1
        // Multiple cores available, as max_counter_uid has to be set to: counter_uid + cores - 1, the maximum
        // allowed value for a counter UID is consequently: uint16_t_max - cores + 1
        if (uid >= std::numeric_limits<uint16_t>::max() - cores + 1)
        {
            throw arm::pipe::ProfilingException("Generating the next UID for profiling would result in an overflow");
        }
        break;
    }
}

} // Anonymous namespace

uint16_t GetNextUid(bool peekOnly)
{
    // The UID used for profiling objects and events. The first valid UID is 1, as 0 is a reserved value
    static uint16_t uid = 1;

    // Check that it is possible to generate the next UID without causing an overflow (throws in case of error)
    ThrowIfCantGenerateNextUid(uid);

    if (peekOnly)
    {
        // Peek only
        return uid;
    }
    else
    {
        // Get the next UID
        return uid++;
    }
}

std::vector<uint16_t> GetNextCounterUids(uint16_t firstUid, uint16_t cores)
{
    // Check that it is possible to generate the next counter UID without causing an overflow (throws in case of error)
    ThrowIfCantGenerateNextUid(firstUid, cores);

    // Get the next counter UIDs
    size_t counterUidsSize = cores == 0 ? 1 : cores;
    std::vector<uint16_t> counterUids(counterUidsSize, 0);
    for (size_t i = 0; i < counterUidsSize; i++)
    {
        counterUids[i] = firstUid++;
    }
    return counterUids;
}

} // namespace pipe
} // namespace arm
