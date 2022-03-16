//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <common/include/CommonProfilingUtils.hpp>

#include <server/include/timelineDecoder/DirectoryCaptureCommandHandler.hpp>

#include <atomic>
#include <iostream>

namespace arm
{

namespace pipe
{

// Utils
uint32_t uint16_t_size = sizeof(uint16_t);
uint32_t uint32_t_size = sizeof(uint32_t);

void DirectoryCaptureCommandHandler::ParseData(const arm::pipe::Packet& packet)
{
    uint16_t categoryRecordCount;
    uint16_t counterSetRecordCount;
    uint16_t deviceRecordCount;

    uint32_t offset = 0;

    if (packet.GetLength() < 8)
    {
        std::cout << "Counter directory packet received." << std::endl;
        return;
    }

    const unsigned char* data = packet.GetData();
    // Body header word 0:
    // 0:15  [16] reserved: all zeros
    offset += uint16_t_size;
    // 16:31 [16] device_records_count: number of entries in the device_records_pointer_table
    deviceRecordCount = ReadUint16(data, offset);
    offset += uint16_t_size;

    // Body header word 1:
    // 0:31 [32] device_records_pointer_table_offset: offset to the device_records_pointer_table
    // The offset is always zero here, as the device record pointer table field is always the first item in the pool
    const uint32_t deviceRecordsPointerTableOffset = ReadUint32(data, offset);
    offset += uint32_t_size;

    // Body header word 2:
    // 0:15  [16] reserved: all zeros
    offset += uint16_t_size;
    // 16:31 [16] counter_set_count: number of entries in the counter_set_pointer_table
    counterSetRecordCount = ReadUint16(data, offset);
    offset += uint16_t_size;

    // Body header word 3:
    // 0:31 [32] counter_set_pointer_table_offset: offset to the counter_set_pointer_table
    const uint32_t counterPointerTableSetOffset = ReadUint32(data, offset);
    offset += uint32_t_size;

    // Body header word 4:
    // 0:15  [16] reserved: all zeros
    offset += uint16_t_size;
    // 16:31 [16] categories_count: number of entries in the categories_pointer_table
    categoryRecordCount = ReadUint16(data, offset);
    offset += uint16_t_size;

    // Body header word 5:
    // 0:31 [32] categories_pointer_table_offset: offset to the categories_pointer_table
    const uint32_t categoriesPointerTableOffset = ReadUint32(data, offset);
    offset += uint32_t_size;

    std::vector<uint32_t> deviceRecordOffsets(deviceRecordCount);
    std::vector<uint32_t> counterSetOffsets(counterSetRecordCount);
    std::vector<uint32_t> categoryOffsets(categoryRecordCount);

    offset = deviceRecordsPointerTableOffset;
    for (uint32_t i = 0; i < deviceRecordCount; ++i)
    {
        deviceRecordOffsets[i] = ReadUint32(data, offset);
        offset += uint32_t_size;
    }

    offset = counterPointerTableSetOffset;
    for (uint32_t i = 0; i < counterSetRecordCount; ++i)
    {
        counterSetOffsets[i] = ReadUint32(data, offset);
        offset += uint32_t_size;
    }

    offset = categoriesPointerTableOffset;
    for (uint32_t i = 0; i < categoryRecordCount; ++i)
    {
        categoryOffsets[i] = ReadUint32(data, offset);
        offset += uint32_t_size;
    }

    offset = deviceRecordsPointerTableOffset;
    for (uint32_t deviceIndex = 0; deviceIndex < deviceRecordCount; ++deviceIndex)
    {
        uint32_t deviceRecordOffset = offset + deviceRecordOffsets[deviceIndex];
        // Device record word 0:
        // 0:15  [16] cores: the number of individual streams of counters for one or more cores of some device
        uint16_t deviceCores = ReadUint16(data, deviceRecordOffset);
        // 16:31 [16] deviceUid: the unique identifier for the device
        deviceRecordOffset += uint16_t_size;
        uint16_t deviceUid = ReadUint16(data, deviceRecordOffset);
        deviceRecordOffset += uint16_t_size;

        // Device record word 1:
        // Offset from the beginning of the device record pool to the name field.
        uint32_t nameOffset = ReadUint32(data, deviceRecordOffset);

        deviceRecordOffset = deviceRecordsPointerTableOffset + nameOffset;

        const std::string& deviceName             = GetStringNameFromBuffer(data, deviceRecordOffset);
        const Device* registeredDevice            = m_CounterDirectory.RegisterDevice(deviceName, deviceCores);
        m_UidTranslation[registeredDevice->m_Uid] = deviceUid;
    }

    offset = counterPointerTableSetOffset;
    for (uint32_t counterSetIndex = 0; counterSetIndex < counterSetRecordCount; ++counterSetIndex)
    {
        uint32_t counterSetOffset = offset + counterSetOffsets[counterSetIndex];

        // Counter set record word 0:
        // 0:15  [16] count: the number of counters which can be active in this set at any one time
        uint16_t counterSetCount = ReadUint16(data, counterSetOffset);
        counterSetOffset += uint16_t_size;

        // 16:31 [16] deviceUid: the unique identifier for the counter_set
        uint16_t counterSetUid = ReadUint16(data, counterSetOffset);
        counterSetOffset += uint16_t_size;

        // Counter set record word 1:
        // 0:31 [32] name_offset: offset from the beginning of the counter set pool to the name field
        // The offset is always zero here, as the name field is always the first (and only) item in the pool
        counterSetOffset += uint32_t_size;
        counterSetOffset += uint32_t_size;

        auto counterSet =
            m_CounterDirectory.RegisterCounterSet(GetStringNameFromBuffer(data, counterSetOffset), counterSetCount);
        m_UidTranslation[counterSet->m_Uid] = counterSetUid;
    }
    ReadCategoryRecords(data, categoriesPointerTableOffset, categoryOffsets);
}

void DirectoryCaptureCommandHandler::ReadCategoryRecords(const unsigned char* const data,
                                                         uint32_t offset,
                                                         std::vector<uint32_t> categoryOffsets)
{
    uint32_t categoryRecordCount = static_cast<uint32_t>(categoryOffsets.size());

    for (uint32_t categoryIndex = 0; categoryIndex < categoryRecordCount; ++categoryIndex)
    {
        uint32_t categoryRecordOffset = offset + categoryOffsets[categoryIndex];

        // Category record word 1:
        // 0:15 Reserved, value 0x0000.
        categoryRecordOffset += uint16_t_size;
        // 16:31 Number of events belonging to this category.
        uint32_t eventCount = ReadUint16(data, categoryRecordOffset);
        categoryRecordOffset += uint16_t_size;

        // Category record word 2
        // 0:31  Offset from the beginning of the category data pool to the event_pointer_table
        uint32_t eventPointerTableOffset = ReadUint32(data, categoryRecordOffset);
        categoryRecordOffset += uint32_t_size;

        // Category record word 3
        // 0:31 Offset from the beginning of the category data pool to the name field.
        uint32_t nameOffset = ReadUint32(data, categoryRecordOffset);
        categoryRecordOffset += uint32_t_size;

        std::vector<uint32_t> eventRecordsOffsets(eventCount);

        eventPointerTableOffset += offset + categoryOffsets[categoryIndex];

        for (uint32_t eventIndex = 0; eventIndex < eventCount; ++eventIndex)
        {
            eventRecordsOffsets[eventIndex] =
                ReadUint32(data, eventPointerTableOffset + uint32_t_size * eventIndex);
        }

        const std::vector<CounterDirectoryEventRecord>& eventRecords =
            ReadEventRecords(data, eventPointerTableOffset, eventRecordsOffsets);

        const Category* category = m_CounterDirectory.RegisterCategory(
            GetStringNameFromBuffer(data, offset + categoryOffsets[categoryIndex] + nameOffset + uint32_t_size));
        for (auto& counter : eventRecords)
        {
            const Counter* registeredCounter = m_CounterDirectory.RegisterCounter(m_ApplicationName,
                                                                                  counter.m_CounterUid,
                                                                                  category->m_Name,
                                                                                  counter.m_CounterClass,
                                                                                  counter.m_CounterInterpolation,
                                                                                  counter.m_CounterMultiplier,
                                                                                  counter.m_CounterName,
                                                                                  counter.m_CounterDescription,
                                                                                  counter.m_CounterUnits);
            m_UidTranslation[registeredCounter->m_Uid] = counter.m_CounterUid;
        }
    }
}

std::vector<CounterDirectoryEventRecord> DirectoryCaptureCommandHandler::ReadEventRecords(
    const unsigned char* data, uint32_t offset, std::vector<uint32_t> eventRecordsOffsets)
{
    uint32_t eventCount = static_cast<uint32_t>(eventRecordsOffsets.size());

    std::vector<CounterDirectoryEventRecord> eventRecords(eventCount);
    for (unsigned long i = 0; i < eventCount; ++i)
    {
        uint32_t eventRecordOffset = eventRecordsOffsets[i] + offset;

        // Event record word 0:
        // 0:15  [16] count_uid: unique ID for the counter. Must be unique across all counters in all categories
        eventRecords[i].m_CounterUid = ReadUint16(data, eventRecordOffset);
        eventRecordOffset += uint16_t_size;
        // 16:31 [16] max_counter_uid: if the device this event is associated with has more than one core and there
        //                             is one of these counters per core this value will be set to
        //                             (counter_uid + cores (from device_record)) - 1.
        //                             If there is only a single core then this value will be the same as
        //                             the counter_uid value
        eventRecords[i].m_MaxCounterUid = ReadUint16(data, eventRecordOffset);
        eventRecordOffset += uint16_t_size;

        // Event record word 1:
        // 0:15  [16] counter_set: UID of the counter_set this event is associated with. Set to zero if the event
        //                         is NOT associated with a counter_set
        eventRecords[i].m_CounterSetUid  = ReadUint16(data, eventRecordOffset);
        eventRecordOffset += uint16_t_size;

        // 16:31 [16] device: UID of the device this event is associated with. Set to zero if the event is NOT
        //                    associated with a device
        eventRecords[i].m_DeviceUid = ReadUint16(data, eventRecordOffset);
        eventRecordOffset += uint16_t_size;

        // Event record word 2:
        // 0:15  [16] interpolation: type describing how to interpolate each data point in a stream of data points
        eventRecords[i].m_CounterInterpolation = ReadUint16(data, eventRecordOffset);
        eventRecordOffset += uint16_t_size;

        // 16:31 [16] class: type describing how to treat each data point in a stream of data points
        eventRecords[i].m_CounterClass = ReadUint16(data, eventRecordOffset);
        eventRecordOffset += uint16_t_size;

        // Event record word 3-4:
        // 0:63 [64] multiplier: internal data stream is represented as integer values, this allows scaling of
        //                       those values as if they are fixed point numbers. Zero is not a valid value
        uint32_t multiplier[2] = { 0u, 0u };

        multiplier[0] = ReadUint32(data, eventRecordOffset);
        eventRecordOffset += uint32_t_size;
        multiplier[1] = ReadUint32(data, eventRecordOffset);
        eventRecordOffset += uint32_t_size;

        std::memcpy(&eventRecords[i].m_CounterMultiplier, &multiplier, sizeof(multiplier));

        // Event record word 5:
        // 0:31 [32] name_eventRecordOffset: eventRecordOffset from the
        // beginning of the event record pool to the name field
        // The eventRecordOffset is always zero here, as the name field is always the first item in the pool
        uint32_t nameOffset = ReadUint32(data, eventRecordOffset);
        eventRecordOffset += uint32_t_size;

        // Event record word 6:
        // 0:31 [32] description_eventRecordOffset: eventRecordOffset from the
        // beginning of the event record pool to the description field
        // The size of the name buffer in bytes
        uint32_t descriptionOffset = ReadUint32(data, eventRecordOffset);
        eventRecordOffset += uint32_t_size;

        // Event record word 7:
        // 0:31 [32] units_eventRecordOffset: (optional) eventRecordOffset from the
        // beginning of the event record pool to the units field.
        // An eventRecordOffset value of zero indicates this field is not provided
        uint32_t unitsOffset = ReadUint32(data, eventRecordOffset);

        eventRecords[i].m_CounterName = GetStringNameFromBuffer(data, offset +
                                                                      eventRecordsOffsets[i] +
                                                                      nameOffset +
                                                                      uint32_t_size);

        eventRecords[i].m_CounterDescription = GetStringNameFromBuffer(data, offset +
                                                                             eventRecordsOffsets[i] +
                                                                             descriptionOffset +
                                                                             uint32_t_size);

        eventRecords[i].m_CounterUnits = unitsOffset == 0 ? arm::pipe::Optional<std::string>() :
                GetStringNameFromBuffer(data, eventRecordsOffsets[i] + offset + unitsOffset + uint32_t_size);
    }

    return eventRecords;
}

void DirectoryCaptureCommandHandler::operator()(const arm::pipe::Packet& packet)
{
    if (!m_QuietOperation)    // Are we supposed to print to stdout?
    {
        std::cout << "Counter directory packet received." << std::endl;
    }

    // The ArmNN counter directory is static per ArmNN instance. Ensure we don't parse it a second time.
    if (!ParsedCounterDirectory())
    {
        ParseData(packet);
        m_AlreadyParsed = true;
    }

    if (!m_QuietOperation)
    {
        PrintCounterDirectory(m_CounterDirectory);
    }
}

const ICounterDirectory& DirectoryCaptureCommandHandler::GetCounterDirectory() const
{
    return m_CounterDirectory;
}

std::string DirectoryCaptureCommandHandler::GetStringNameFromBuffer(const unsigned char* const data, uint32_t offset)
{
    std::string deviceName;
    uint8_t nextChar = ReadUint8(data, offset);

    while (isprint(nextChar))
    {
        deviceName += static_cast<char>(nextChar);
        offset++;
        nextChar = ReadUint8(data, offset);
    }

    return deviceName;
}

}    // namespace pipe

}    // namespace arm
