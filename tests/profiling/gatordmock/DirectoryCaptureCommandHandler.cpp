//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <atomic>
#include "DirectoryCaptureCommandHandler.hpp"

namespace armnn
{

namespace gatordmock
{

// Utils
uint32_t uint16_t_size = sizeof(uint16_t);
uint32_t uint32_t_size = sizeof(uint32_t);

void DirectoryCaptureCommandHandler::ParseData(const armnn::profiling::Packet& packet)
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

    const unsigned char* data = reinterpret_cast<const unsigned char*>(packet.GetData());
    // Body header word 0:
    // 0:15  [16] reserved: all zeros
    offset += uint16_t_size;
    // 16:31 [16] device_records_count: number of entries in the device_records_pointer_table
    deviceRecordCount = profiling::ReadUint16(data, offset);
    offset += uint16_t_size;

    // Body header word 1:
    // 0:31 [32] device_records_pointer_table_offset: offset to the device_records_pointer_table
    // The offset is always zero here, as the device record pointer table field is always the first item in the pool
    offset += uint32_t_size;

    // Body header word 2:
    // 0:15  [16] reserved: all zeros
    offset += uint16_t_size;
    // 16:31 [16] counter_set_count: number of entries in the counter_set_pointer_table
    counterSetRecordCount = profiling::ReadUint16(data, offset);
    offset += uint16_t_size;

    // Body header word 3:
    // 0:31 [32] counter_set_pointer_table_offset: offset to the counter_set_pointer_table
    // counterPointerTableSetOffset = profiling::ReadUint32(data, offset);
    offset += uint32_t_size;

    // Body header word 4:
    // 0:15  [16] reserved: all zeros
    offset += uint16_t_size;
    // 16:31 [16] categories_count: number of entries in the categories_pointer_table
    categoryRecordCount = profiling::ReadUint16(data, offset);
    offset += uint16_t_size;

    // Body header word 5:
    // 0:31 [32] categories_pointer_table_offset: offset to the categories_pointer_table
    // categoriesPointerTableOffset = profiling::ReadUint32(data, offset);
    offset += uint32_t_size;

    std::vector<uint32_t> deviceRecordOffsets(deviceRecordCount);
    std::vector<uint32_t> counterSetOffsets(counterSetRecordCount);
    std::vector<uint32_t> categoryOffsets(categoryRecordCount);

    for (uint32_t i = 0; i < deviceRecordCount; ++i)
    {
        deviceRecordOffsets[i] = profiling::ReadUint32(data, offset);
        offset += uint32_t_size;
    }

    for (uint32_t i = 0; i < counterSetRecordCount; ++i)
    {
        counterSetOffsets[i] = profiling::ReadUint32(data, offset);
        offset += uint32_t_size;
    }

    for (uint32_t i = 0; i < categoryRecordCount; ++i)
    {
        categoryOffsets[i] = profiling::ReadUint32(data, offset);
        offset += uint32_t_size;
    }

    m_CounterDirectory.m_DeviceRecords = ReadDeviceRecords(data, offset, deviceRecordOffsets);
    m_CounterDirectory.m_CounterSets   = ReadCounterSetRecords(data, offset, counterSetOffsets);
    m_CounterDirectory.m_Categories    = ReadCategoryRecords(data, offset, categoryOffsets);

    m_CounterDirectoryCount.operator++(std::memory_order_release);
}

std::vector<DeviceRecord> DirectoryCaptureCommandHandler::ReadDeviceRecords(const unsigned char* const data,
                                                                            uint32_t offset,
                                                                            std::vector<uint32_t> deviceRecordOffsets)
{
    uint32_t deviceRecordCount = static_cast<uint32_t >(deviceRecordOffsets.size());
    std::vector<DeviceRecord> deviceRecords(deviceRecordCount);

    for(uint32_t deviceIndex = 0; deviceIndex < deviceRecordCount; ++deviceIndex)
    {
        uint32_t deviceRecordOffset = offset + deviceRecordOffsets[deviceIndex];
        // Device record word 0:
        // 0:15  [16] cores: the number of individual streams of counters for one or more cores of some device
        deviceRecords[deviceIndex].m_DeviceCores = profiling::ReadUint16(data, deviceRecordOffset);
        // 16:31 [16] deviceUid: the unique identifier for the device
        deviceRecordOffset += uint16_t_size;
        deviceRecords[deviceIndex].m_DeviceUid = profiling::ReadUint16(data, deviceRecordOffset);
        deviceRecordOffset += uint16_t_size;

        // Device record word 1:
        // Offset from the beginning of the device record pool to the name field.
        uint32_t nameOffset = profiling::ReadUint32(data, deviceRecordOffset);

        deviceRecordOffset += uint32_t_size;
        deviceRecordOffset += uint32_t_size;
        deviceRecordOffset += nameOffset;

        deviceRecords[deviceIndex].m_DeviceName = GetStringNameFromBuffer(data, deviceRecordOffset);
    }

    return  deviceRecords;
}


std::vector<CounterSetRecord>
        DirectoryCaptureCommandHandler::ReadCounterSetRecords(const unsigned char* const data,
                                                              uint32_t offset,
                                                              std::vector<uint32_t> counterSetOffsets)
{
    uint32_t counterSetRecordCount = static_cast<uint32_t >(counterSetOffsets.size());
    std::vector<CounterSetRecord> counterSets(counterSetRecordCount);

    for (uint32_t counterSetIndex = 0; counterSetIndex < counterSetRecordCount; ++counterSetIndex)
    {
        uint32_t counterSetOffset = offset + counterSetOffsets[counterSetIndex];

        // Counter set record word 0:
        // 0:15  [16] count: the number of counters which can be active in this set at any one time
        counterSets[counterSetIndex].m_CounterSetCount = profiling::ReadUint16(data, counterSetOffset);
        counterSetOffset += uint16_t_size;

        // 16:31 [16] deviceUid: the unique identifier for the counter_set
        counterSets[counterSetIndex].m_CounterSetUid = profiling::ReadUint16(data, counterSetOffset);
        counterSetOffset += uint16_t_size;

        // Counter set record word 1:
        // 0:31 [32] name_offset: offset from the beginning of the counter set pool to the name field
        // The offset is always zero here, as the name field is always the first (and only) item in the pool
        counterSetOffset += uint32_t_size;
        counterSetOffset += uint32_t_size;

        counterSets[counterSetIndex].m_CounterSetName = GetStringNameFromBuffer(data, counterSetOffset);
    }

    return counterSets;
}

std::vector<CategoryRecord> DirectoryCaptureCommandHandler::ReadCategoryRecords(const unsigned char* const data,
                                                                                uint32_t offset,
                                                                                std::vector<uint32_t> categoryOffsets)
{
    uint32_t categoryRecordCount = static_cast<uint32_t >(categoryOffsets.size());
    std::vector<CategoryRecord> categories(categoryRecordCount);

    for (uint32_t categoryIndex = 0; categoryIndex < categoryRecordCount; ++categoryIndex)
    {
        uint32_t categoryRecordOffset = offset + categoryOffsets[categoryIndex];

        // Category record word 0:
        // 0:15  The deviceUid of a counter_set the category is associated with.
        // Set to zero if the category is NOT associated with a counter set.
        categories[categoryIndex].m_CounterSet = profiling::ReadUint16(data, categoryRecordOffset);
        categoryRecordOffset += uint16_t_size;

        // 16:31 The deviceUid of a device element which identifies some hardware device that the category belongs to.
        // Set to zero if the category is NOT associated with a device
        categories[categoryIndex].m_DeviceUid = profiling::ReadUint16(data, categoryRecordOffset);
        categoryRecordOffset += uint16_t_size;

        // Category record word 1:
        // 0:15 Reserved, value 0x0000.
        categoryRecordOffset += uint16_t_size;
        // 16:31 Number of events belonging to this category.
        categories[categoryIndex].m_EventCount = profiling::ReadUint16(data, categoryRecordOffset);
        categoryRecordOffset += uint16_t_size;

        // Category record word 2
        // 0:31  Offset from the beginning of the category data pool to the event_pointer_table
        uint32_t eventPointerTableOffset = profiling::ReadUint32(data, categoryRecordOffset);
        categoryRecordOffset += uint32_t_size;

        // Category record word 3
        // 0:31 Offset from the beginning of the category data pool to the name field.
        uint32_t nameOffset = profiling::ReadUint32(data, categoryRecordOffset);
        categoryRecordOffset += uint32_t_size;

        //Get the events for the category
        uint32_t eventCount = categories[categoryIndex].m_EventCount;

        std::vector<uint32_t> eventRecordsOffsets(eventCount);

        eventPointerTableOffset += categoryRecordOffset;

        for (uint32_t eventIndex = 0; eventIndex < eventCount; ++eventIndex)
        {
            eventRecordsOffsets[eventIndex] =
                    profiling::ReadUint32(data, eventPointerTableOffset + uint32_t_size * eventIndex);
        }

        categories[categoryIndex].m_EventRecords = ReadEventRecords(data, categoryRecordOffset, eventRecordsOffsets);

        categoryRecordOffset += uint32_t_size;

        categories[categoryIndex].m_CategoryName = GetStringNameFromBuffer(data, categoryRecordOffset + nameOffset);
    }

    return categories;
}


std::vector<EventRecord> DirectoryCaptureCommandHandler::ReadEventRecords(const unsigned char* const data,
                                                                          uint32_t offset,
                                                                          std::vector<uint32_t> eventRecordsOffsets)
{
    uint32_t eventCount = static_cast<uint32_t>(eventRecordsOffsets.size());

    std::vector<EventRecord> eventRecords(eventCount);
    for (unsigned long i = 0; i < eventCount; ++i)
    {
        uint32_t eventRecordOffset = eventRecordsOffsets[i] + offset;

        // Event record word 0:
        // 0:15  [16] count_uid: unique ID for the counter. Must be unique across all counters in all categories
        eventRecords[i].m_CounterUid = profiling::ReadUint16(data, eventRecordOffset);
        eventRecordOffset += uint16_t_size;
        // 16:31 [16] max_counter_uid: if the device this event is associated with has more than one core and there
        //                             is one of these counters per core this value will be set to
        //                             (counter_uid + cores (from device_record)) - 1.
        //                             If there is only a single core then this value will be the same as
        //                             the counter_uid value
        eventRecords[i].m_MaxCounterUid = profiling::ReadUint16(data, eventRecordOffset);
        eventRecordOffset += uint16_t_size;

        // Event record word 1:
        // 0:15  [16] counter_set: UID of the counter_set this event is associated with. Set to zero if the event
        //                         is NOT associated with a counter_set
        eventRecords[i].m_DeviceUid = profiling::ReadUint16(data, eventRecordOffset);
        eventRecordOffset += uint16_t_size;

        // 16:31 [16] device: UID of the device this event is associated with. Set to zero if the event is NOT
        //                    associated with a device
        eventRecords[i].m_CounterSetUid = profiling::ReadUint16(data, eventRecordOffset);
        eventRecordOffset += uint16_t_size;

        // Event record word 2:
        // 0:15  [16] interpolation: type describing how to interpolate each data point in a stream of data points
        eventRecords[i].m_CounterClass =profiling::ReadUint16(data, eventRecordOffset);
        eventRecordOffset += uint16_t_size;

        // 16:31 [16] class: type describing how to treat each data point in a stream of data points
        eventRecords[i].m_CounterInterpolation = profiling::ReadUint16(data, eventRecordOffset);
        eventRecordOffset += uint16_t_size;

        // Event record word 3-4:
        // 0:63 [64] multiplier: internal data stream is represented as integer values, this allows scaling of
        //                       those values as if they are fixed point numbers. Zero is not a valid value
        uint32_t multiplier[2] = { 0u, 0u };

        multiplier[0] = profiling::ReadUint32(data, eventRecordOffset);
        eventRecordOffset += uint32_t_size;
        multiplier[1] = profiling::ReadUint32(data, eventRecordOffset);
        eventRecordOffset += uint32_t_size;

        std::memcpy(&eventRecords[i].m_CounterMultiplier, &multiplier, sizeof(multiplier));

        // Event record word 5:
        // 0:31 [32] name_eventRecordOffset: eventRecordOffset from the
        // beginning of the event record pool to the name field
        // The eventRecordOffset is always zero here, as the name field is always the first item in the pool
        eventRecordOffset += uint32_t_size;

        // Event record word 6:
        // 0:31 [32] description_eventRecordOffset: eventRecordOffset from the
        // beginning of the event record pool to the description field
        // The size of the name buffer in bytes
        uint32_t descriptionOffset = profiling::ReadUint32(data, eventRecordOffset);
        eventRecordOffset += uint32_t_size;

        // Event record word 7:
        // 0:31 [32] units_eventRecordOffset: (optional) eventRecordOffset from the
        // beginning of the event record pool to the units field.
        // An eventRecordOffset value of zero indicates this field is not provided
        uint32_t unitsOffset = profiling::ReadUint32(data, eventRecordOffset);
        eventRecordOffset += uint32_t_size;
        eventRecordOffset += uint32_t_size;

        eventRecords[i].m_CounterName = GetStringNameFromBuffer(data, eventRecordOffset);

        eventRecords[i].m_CounterDescription = GetStringNameFromBuffer(data, eventRecordOffset + descriptionOffset);

        eventRecords[i].m_CounterUnits = GetStringNameFromBuffer(data, eventRecordOffset + unitsOffset);
    }

    return eventRecords;
}

void DirectoryCaptureCommandHandler::operator()(const profiling::Packet& packet)
{
    if (!m_QuietOperation)// Are we supposed to print to stdout?
    {
        std::cout << "Counter directory packet received." << std::endl;
    }

    ParseData(packet);

    if (!m_QuietOperation)
    {
        m_CounterDirectory.print();
    }
}

CounterDirectory DirectoryCaptureCommandHandler::GetCounterDirectory() const
{
    return m_CounterDirectory;
}

uint32_t DirectoryCaptureCommandHandler::GetCounterDirectoryCount() const
{
    return m_CounterDirectoryCount.load(std::memory_order_acquire);
}

} // namespace gatordmock

} // namespace armnn