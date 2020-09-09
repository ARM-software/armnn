//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SendCounterPacket.hpp"
#include <common/include/EncodeVersion.hpp>

#include <armnn/Exceptions.hpp>
#include <armnn/Conversion.hpp>
#include <Processes.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <common/include/Constants.hpp>
#include <common/include/SwTrace.hpp>

#include <fmt/format.h>

#include <cstring>

namespace armnn
{

namespace profiling
{

void SendCounterPacket::SendStreamMetaDataPacket()
{
    const std::string info(GetSoftwareInfo());
    const std::string hardwareVersion(GetHardwareVersion());
    const std::string softwareVersion(GetSoftwareVersion());
    const std::string processName = GetProcessName().substr(0, 60);

    const uint32_t infoSize =            armnn::numeric_cast<uint32_t>(info.size()) + 1;
    const uint32_t hardwareVersionSize = armnn::numeric_cast<uint32_t>(hardwareVersion.size()) + 1;
    const uint32_t softwareVersionSize = armnn::numeric_cast<uint32_t>(softwareVersion.size()) + 1;
    const uint32_t processNameSize =     armnn::numeric_cast<uint32_t>(processName.size()) + 1;

    const uint32_t sizeUint32 = sizeof(uint32_t);

    const uint32_t headerSize = 2 * sizeUint32;
    const uint32_t bodySize = 10 * sizeUint32;
    const uint32_t packetVersionCountSize = sizeUint32;

    // Supported Packets
    // Packet Encoding version 1.0.0
    // Control packet family
    //   Stream metadata packet (packet family=0; packet id=0)
    //   Connection Acknowledged packet ( packet family=0, packet id=1) Version 1.0.0
    //   Counter Directory packet (packet family=0; packet id=2) Version 1.0.0
    //   Request Counter Directory packet ( packet family=0, packet id=3) Version 1.0.0
    //   Periodic Counter Selection packet ( packet family=0, packet id=4) Version 1.0.0
    //   Per Job Counter Selection packet ( packet family=0, packet id=5) Version 1.0.0
    //   Activate Timeline Reporting (packet family = 0, packet id = 6) Version 1.0.0
    //   Deactivate Timeline Reporting (packet family = 0, packet id = 7) Version 1.0.0
    // Counter Packet Family
    //   Periodic Counter Capture (packet_family = 3, packet_class = 0, packet_type = 0) Version 1.0.0
    //   Per-Job Counter Capture (packet_family = 3, packet_class = 1, packet_type = 0,1) Version  1.0.0
    // Timeline Packet Family
    //   Timeline Message Directory (packet_family = 1, packet_class = 0, packet_type = 0) Version 1.0.0
    //   Timeline Message (packet_family = 1, packet_class = 0, packet_type = 1) Version 1.0.0
    std::vector<std::pair<uint32_t, uint32_t>> packetVersions;
    packetVersions.push_back(std::make_pair(ConstructHeader(0, 0), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(0, 1), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(0, 2), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(0, 3), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(0, 4), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(0, 5), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(0, 6), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(0, 7), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(3, 0, 0), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(3, 1, 0), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(3, 1, 1), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(1, 0, 0), arm::pipe::EncodeVersion(1, 0, 0)));
    packetVersions.push_back(std::make_pair(ConstructHeader(1, 0, 1), arm::pipe::EncodeVersion(1, 0, 0)));
    uint32_t numberOfVersions = armnn::numeric_cast<uint32_t>(packetVersions.size());
    uint32_t packetVersionSize = armnn::numeric_cast<uint32_t>(numberOfVersions * 2 * sizeUint32);

    const uint32_t payloadSize = armnn::numeric_cast<uint32_t>(infoSize + hardwareVersionSize +
                                                               softwareVersionSize + processNameSize +
                                                               packetVersionCountSize + packetVersionSize);

    const uint32_t totalSize = headerSize + bodySize + payloadSize;
    uint32_t offset = 0;
    uint32_t reserved = 0;

    IPacketBufferPtr writeBuffer = m_BufferManager.Reserve(totalSize, reserved);

    if (writeBuffer == nullptr || reserved < totalSize)
    {
        CancelOperationAndThrow<BufferExhaustion>(
            writeBuffer,
            fmt::format("No space left in buffer. Unable to reserve ({}) bytes.", totalSize));
    }

    try
    {
        // Create header

        WriteUint32(writeBuffer, offset, 0);
        offset += sizeUint32;
        WriteUint32(writeBuffer, offset, totalSize - headerSize);

        // Packet body

        offset += sizeUint32;
        WriteUint32(writeBuffer, offset, arm::pipe::PIPE_MAGIC); // pipe_magic
        offset += sizeUint32;
        WriteUint32(writeBuffer, offset, arm::pipe::EncodeVersion(1, 0, 0)); // stream_metadata_version
        offset += sizeUint32;
        WriteUint32(writeBuffer, offset, MAX_METADATA_PACKET_LENGTH); // max_data_length
        offset += sizeUint32;
        int pid = armnnUtils::Processes::GetCurrentId();
        WriteUint32(writeBuffer, offset, armnn::numeric_cast<uint32_t>(pid)); // pid
        offset += sizeUint32;
        uint32_t poolOffset = bodySize;
        WriteUint32(writeBuffer, offset, poolOffset); // offset_info
        offset += sizeUint32;
        poolOffset += infoSize;
        WriteUint32(writeBuffer, offset, poolOffset); // offset_hw_version
        offset += sizeUint32;
        poolOffset += hardwareVersionSize;
        WriteUint32(writeBuffer, offset, poolOffset); // offset_sw_version
        offset += sizeUint32;
        poolOffset += softwareVersionSize;
        WriteUint32(writeBuffer, offset, poolOffset); // offset_process_name
        offset += sizeUint32;
        poolOffset += processNameSize;
        WriteUint32(writeBuffer, offset, poolOffset); // offset_packet_version_table
        offset += sizeUint32;
        WriteUint32(writeBuffer, offset, 0); // reserved
        offset += sizeUint32;

        // Pool

        if (infoSize)
        {
            memcpy(&writeBuffer->GetWritableData()[offset], info.c_str(), infoSize);
            offset += infoSize;
        }

        memcpy(&writeBuffer->GetWritableData()[offset], hardwareVersion.c_str(), hardwareVersionSize);
        offset += hardwareVersionSize;
        memcpy(&writeBuffer->GetWritableData()[offset], softwareVersion.c_str(), softwareVersionSize);
        offset += softwareVersionSize;
        memcpy(&writeBuffer->GetWritableData()[offset], processName.c_str(), processNameSize);
        offset += processNameSize;

        if (!packetVersions.empty())
        {
            // Packet Version Count
            WriteUint32(writeBuffer, offset, numberOfVersions << 16);
            offset += sizeUint32;

            // Packet Version Entries
            for (std::pair<uint32_t, uint32_t>& packetVersion : packetVersions)
            {
                WriteUint32(writeBuffer, offset, packetVersion.first);
                offset += sizeUint32;
                WriteUint32(writeBuffer, offset, packetVersion.second);
                offset += sizeUint32;
            }
        }
    }
    catch(...)
    {
        CancelOperationAndThrow<RuntimeException>(writeBuffer, "Error processing packet.");
    }

    m_BufferManager.Commit(writeBuffer, totalSize, false);
}

bool SendCounterPacket::CreateCategoryRecord(const CategoryPtr& category,
                                             const Counters& counters,
                                             CategoryRecord& categoryRecord,
                                             std::string& errorMessage)
{
    ARMNN_ASSERT(category);

    const std::string& categoryName = category->m_Name;
    ARMNN_ASSERT(!categoryName.empty());

    // Remove any duplicate counters
    std::vector<uint16_t> categoryCounters;
    for (size_t counterIndex = 0; counterIndex < category->m_Counters.size(); ++counterIndex)
    {
        uint16_t counterUid = category->m_Counters.at(counterIndex);
        auto it = counters.find(counterUid);
        if (it == counters.end())
        {
            errorMessage = fmt::format("Counter ({}) not found in category ({})",
                                       counterUid,
                                       category->m_Name );
            return false;
        }

        const CounterPtr& counter = it->second;

        if (counterUid == counter->m_MaxCounterUid)
        {
            categoryCounters.emplace_back(counterUid);
        }
    }
    if (categoryCounters.empty())
    {
        errorMessage = fmt::format("No valid counters found in category ({})", categoryName);
        return false;
    }

    // Utils
    const size_t uint32_t_size = sizeof(uint32_t);

    // Convert the device name into a SWTrace namestring
    std::vector<uint32_t> categoryNameBuffer;
    if (!arm::pipe::StringToSwTraceString<arm::pipe::SwTraceNameCharPolicy>(categoryName, categoryNameBuffer))
    {
        errorMessage = fmt::format("Cannot convert the name of category ({}) to an SWTrace namestring",
                                   categoryName);
        return false;
    }

    // Category record word 1:
    // 16:31 [16] event_count: number of events belonging to this category
    // 0:15  [16] reserved: all zeros
    const uint32_t categoryRecordWord1 = static_cast<uint32_t>(categoryCounters.size()) << 16;

    // Category record word 2:
    // 0:31 [32] event_pointer_table_offset: offset from the beginning of the category data pool to
    //                                       the event_pointer_table
    const uint32_t categoryRecordWord2 = static_cast<uint32_t>(3u * uint32_t_size);

    // Process the event records
    const size_t counterCount = categoryCounters.size();
    std::vector<EventRecord> eventRecords(counterCount);
    std::vector<uint32_t> eventRecordOffsets(counterCount, 0);
    size_t eventRecordsSize = 0;
    uint32_t eventRecordsOffset = armnn::numeric_cast<uint32_t>(
                    (eventRecords.size() + categoryNameBuffer.size()) * uint32_t_size);
    for (size_t counterIndex = 0, eventRecordIndex = 0, eventRecordOffsetIndex = 0;
         counterIndex < counterCount;
         counterIndex++, eventRecordIndex++, eventRecordOffsetIndex++)
    {
        uint16_t counterUid = categoryCounters.at(counterIndex);
        auto it = counters.find(counterUid);
        const CounterPtr& counter = it->second;

        EventRecord& eventRecord = eventRecords.at(eventRecordIndex);
        if (!CreateEventRecord(counter, eventRecord, errorMessage))
        {
            return false;
        }

        // Update the total size in words of the event records
        eventRecordsSize += eventRecord.size();

        // Add the event record offset to the event pointer table offset field
        eventRecordOffsets[eventRecordOffsetIndex] = eventRecordsOffset;
        eventRecordsOffset += armnn::numeric_cast<uint32_t>(eventRecord.size() * uint32_t_size);
    }

    // Category record word 3:
    // 0:31 [32] name_offset (offset from the beginning of the category data pool to the name field)
    const uint32_t categoryRecordWord3 = armnn::numeric_cast<uint32_t>(
            (3u + eventRecordOffsets.size()) * uint32_t_size);

    // Calculate the size in words of the category record
    const size_t categoryRecordSize = 3u +// The size of the fixed part (device + counter_set + event_count +
                                          // reserved + event_pointer_table_offset + name_offset)
                                      eventRecordOffsets.size() + // The size of the variable part (
                                      categoryNameBuffer.size() + // the event pointer table + the category name
                                      eventRecordsSize;           // including the null-terminator + the event records)

    // Allocate the necessary space for the category record
    categoryRecord.resize(categoryRecordSize);

    ARMNN_NO_CONVERSION_WARN_BEGIN
    // Create the category record
    categoryRecord[0] = categoryRecordWord1; // event_count + reserved
    categoryRecord[1] = categoryRecordWord2; // event_pointer_table_offset
    categoryRecord[2] = categoryRecordWord3; // name_offset
    auto offset = categoryRecord.begin() + 3u;
    std::copy(eventRecordOffsets.begin(), eventRecordOffsets.end(), offset); // event_pointer_table
    offset += eventRecordOffsets.size();
    std::copy(categoryNameBuffer.begin(), categoryNameBuffer.end(), offset); // name
    offset += categoryNameBuffer.size();
    for (const EventRecord& eventRecord : eventRecords)
    {
        std::copy(eventRecord.begin(), eventRecord.end(), offset); // event_record
        offset += eventRecord.size();
    }
    ARMNN_NO_CONVERSION_WARN_END

    return true;
}

bool SendCounterPacket::CreateDeviceRecord(const DevicePtr& device,
                                           DeviceRecord& deviceRecord,
                                           std::string& errorMessage)
{
    ARMNN_ASSERT(device);

    uint16_t deviceUid = device->m_Uid;
    const std::string& deviceName = device->m_Name;
    uint16_t deviceCores = device->m_Cores;

    ARMNN_ASSERT(!deviceName.empty());

    // Device record word 0:
    // 16:31 [16] uid: the unique identifier for the device
    // 0:15  [16] cores: the number of individual streams of counters for one or more cores of some device
    const uint32_t deviceRecordWord0 = (static_cast<uint32_t>(deviceUid) << 16) |
                                 (static_cast<uint32_t>(deviceCores));

    // Device record word 1:
    // 0:31 [32] name_offset: offset from the beginning of the device record pool to the name field
    const uint32_t deviceRecordWord1 = 8u; // The offset is always eight here, as the name field is always
                                           // the first (and only) item in the pool and there are two device words

    // Convert the device name into a SWTrace string
    std::vector<uint32_t> deviceNameBuffer;
    if (!arm::pipe::StringToSwTraceString<arm::pipe::SwTraceCharPolicy>(deviceName, deviceNameBuffer))
    {
        errorMessage = fmt::format("Cannot convert the name of device {} ({}) to an SWTrace string",
                                   deviceUid,
                                   deviceName);
        return false;
    }

    // Calculate the size in words of the device record
    const size_t deviceRecordSize = 2u + // The size of the fixed part (uid + cores + name_offset)
                              deviceNameBuffer.size(); // The size of the variable part (the device name including
                                                       // the null-terminator)

    // Allocate the necessary space for the device record
    deviceRecord.resize(deviceRecordSize);

    // Create the device record
    deviceRecord[0] = deviceRecordWord0; // uid + core
    deviceRecord[1] = deviceRecordWord1; // name_offset
    auto offset = deviceRecord.begin() + 2u;
    std::copy(deviceNameBuffer.begin(), deviceNameBuffer.end(), offset); // name

    return true;
}

bool SendCounterPacket::CreateCounterSetRecord(const CounterSetPtr& counterSet,
                                               CounterSetRecord& counterSetRecord,
                                               std::string& errorMessage)
{
    ARMNN_ASSERT(counterSet);

    uint16_t counterSetUid = counterSet->m_Uid;
    const std::string& counterSetName = counterSet->m_Name;
    uint16_t counterSetCount = counterSet->m_Count;

    ARMNN_ASSERT(!counterSetName.empty());

    // Counter set record word 0:
    // 16:31 [16] uid: the unique identifier for the counter_set
    // 0:15  [16] count: the number of counters which can be active in this set at any one time
    const uint32_t counterSetRecordWord0 = (static_cast<uint32_t>(counterSetUid) << 16) |
                                           (static_cast<uint32_t>(counterSetCount));

    // Counter set record word 1:
    // 0:31 [32] name_offset: offset from the beginning of the counter set pool to the name field
    const uint32_t counterSetRecordWord1 = 8u; // The offset is always eight here, as the name field is always
                                               // the first (and only) item in the pool after the two counter set words

    // Convert the device name into a SWTrace namestring
    std::vector<uint32_t> counterSetNameBuffer;
    if (!arm::pipe::StringToSwTraceString<arm::pipe::SwTraceNameCharPolicy>(counterSet->m_Name, counterSetNameBuffer))
    {
        errorMessage = fmt::format("Cannot convert the name of counter set {} ({}) to an SWTrace namestring",
                                   counterSetUid,
                                   counterSetName);
        return false;
    }

    // Calculate the size in words of the counter set record
    const size_t counterSetRecordSize = 2u + // The size of the fixed part (uid + cores + name_offset)
                                        counterSetNameBuffer.size(); // The size of the variable part (the counter set
                                                                     // name including the null-terminator)

    // Allocate the space for the counter set record
    counterSetRecord.resize(counterSetRecordSize);

    // Create the counter set record
    counterSetRecord[0] = counterSetRecordWord0; // uid + core
    counterSetRecord[1] = counterSetRecordWord1; // name_offset
    auto offset = counterSetRecord.begin() + 2u;
    std::copy(counterSetNameBuffer.begin(), counterSetNameBuffer.end(), offset); // name

    return true;
}

bool SendCounterPacket::CreateEventRecord(const CounterPtr& counter,
                                          EventRecord& eventRecord,
                                          std::string& errorMessage)
{
    ARMNN_ASSERT(counter);

    uint16_t           counterUid           = counter->m_Uid;
    uint16_t           maxCounterUid        = counter->m_MaxCounterUid;
    uint16_t           deviceUid            = counter->m_DeviceUid;
    uint16_t           counterSetUid        = counter->m_CounterSetUid;
    uint16_t           counterClass         = counter->m_Class;
    uint16_t           counterInterpolation = counter->m_Interpolation;
    double             counterMultiplier    = counter->m_Multiplier;
    const std::string& counterName          = counter->m_Name;
    const std::string& counterDescription   = counter->m_Description;
    const std::string& counterUnits         = counter->m_Units;

    ARMNN_ASSERT(counterClass == 0 || counterClass == 1);
    ARMNN_ASSERT(counterInterpolation == 0 || counterInterpolation == 1);
    ARMNN_ASSERT(counterMultiplier);

    // Utils
    const size_t uint32_t_size = sizeof(uint32_t);
    // eventRecordBlockSize is the size of the fixed part
    // (counter_uid + max_counter_uid + device +
    // counter_set + class + interpolation +
    // multiplier + name_offset + description_offset +
    // units_offset)
    const size_t eventRecordBlockSize = 8u;

    // Event record word 0:
    // 16:31 [16] max_counter_uid: if the device this event is associated with has more than one core and there
    //                             is one of these counters per core this value will be set to
    //                             (counter_uid + cores (from device_record)) - 1.
    //                             If there is only a single core then this value will be the same as
    //                             the counter_uid value
    // 0:15  [16] count_uid: unique ID for the counter. Must be unique across all counters in all categories
    const uint32_t eventRecordWord0 = (static_cast<uint32_t>(maxCounterUid) << 16) |
                                      (static_cast<uint32_t>(counterUid));

    // Event record word 1:
    // 16:31 [16] device: UID of the device this event is associated with. Set to zero if the event is NOT
    //                    associated with a device
    // 0:15  [16] counter_set: UID of the counter_set this event is associated with. Set to zero if the event
    //                         is NOT associated with a counter_set
    const uint32_t eventRecordWord1 = (static_cast<uint32_t>(deviceUid) << 16) |
                                      (static_cast<uint32_t>(counterSetUid));

    // Event record word 2:
    // 16:31 [16] class: type describing how to treat each data point in a stream of data points
    // 0:15  [16] interpolation: type describing how to interpolate each data point in a stream of data points
    const uint32_t eventRecordWord2 = (static_cast<uint32_t>(counterClass) << 16) |
                                      (static_cast<uint32_t>(counterInterpolation));

    // Event record word 3-4:
    // 0:63 [64] multiplier: internal data stream is represented as integer values, this allows scaling of
    //                       those values as if they are fixed point numbers. Zero is not a valid value
    uint32_t multiplier[2] = { 0u, 0u };
    ARMNN_ASSERT(sizeof(counterMultiplier) == sizeof(multiplier));
    std::memcpy(multiplier, &counterMultiplier, sizeof(multiplier));
    const uint32_t eventRecordWord3 = multiplier[0];
    const uint32_t eventRecordWord4 = multiplier[1];

    // Event record word 5:
    // 0:31 [32] name_offset: offset from the beginning of the event record pool to the name field
    const uint32_t eventRecordWord5 = static_cast<uint32_t>(eventRecordBlockSize * uint32_t_size);

    // Convert the counter name into a SWTrace string
    std::vector<uint32_t> counterNameBuffer;
    if (!arm::pipe::StringToSwTraceString<arm::pipe::SwTraceCharPolicy>(counterName, counterNameBuffer))
    {
        errorMessage = fmt::format("Cannot convert the name of counter {} (name: {}) to an SWTrace string",
                                   counterUid,
                                   counterName);
        return false;
    }

    // Event record word 6:
    // 0:31 [32] description_offset: offset from the beginning of the event record pool to the description field
    // The size of the name buffer in bytes
    uint32_t eventRecordWord6 =
            static_cast<uint32_t>((counterNameBuffer.size() + eventRecordBlockSize) * uint32_t_size);

    // Convert the counter description into a SWTrace string
    std::vector<uint32_t> counterDescriptionBuffer;
    if (!arm::pipe::StringToSwTraceString<arm::pipe::SwTraceCharPolicy>(counterDescription, counterDescriptionBuffer))
    {
        errorMessage = fmt::format("Cannot convert the description of counter {} (description: {}) "
                                   "to an SWTrace string",
                                   counterUid,
                                   counterName);
        return false;
    }

    // Event record word 7:
    // 0:31 [32] units_offset: (optional) offset from the beginning of the event record pool to the units field.
    //                         An offset value of zero indicates this field is not provided
    bool includeUnits = !counterUnits.empty();
    // The size of the description buffer in bytes
    const uint32_t eventRecordWord7 = includeUnits ?
                                eventRecordWord6 +
                                armnn::numeric_cast<uint32_t>(counterDescriptionBuffer.size()
                                * uint32_t_size) :
                                0;

    // Convert the counter units into a SWTrace namestring (optional)
    std::vector<uint32_t> counterUnitsBuffer;
    if (includeUnits)
    {
        // Convert the counter units into a SWTrace namestring
        if (!arm::pipe::StringToSwTraceString<arm::pipe::SwTraceNameCharPolicy>(counterUnits, counterUnitsBuffer))
        {
            errorMessage = fmt::format("Cannot convert the units of counter {} (units: {}) to an SWTrace string",
                                       counterUid,
                                       counterName);
            return false;
        }
    }

    // Calculate the size in words of the event record
    const size_t eventRecordSize = eventRecordBlockSize +
                                   counterNameBuffer.size() +        // The size of the variable part (the counter name,
                                   counterDescriptionBuffer.size() + // description and units
                                   counterUnitsBuffer.size();        // including the null-terminator)

    // Allocate the space for the event record
    eventRecord.resize(eventRecordSize);

    ARMNN_NO_CONVERSION_WARN_BEGIN
    // Create the event record
    eventRecord[0] = eventRecordWord0; // max_counter_uid + counter_uid
    eventRecord[1] = eventRecordWord1; // device + counter_set
    eventRecord[2] = eventRecordWord2; // class + interpolation
    eventRecord[3] = eventRecordWord3; // multiplier
    eventRecord[4] = eventRecordWord4; // multiplier
    eventRecord[5] = eventRecordWord5; // name_offset
    eventRecord[6] = eventRecordWord6; // description_offset
    eventRecord[7] = eventRecordWord7; // units_offset
    auto offset = eventRecord.begin() + 8u;
    std::copy(counterNameBuffer.begin(), counterNameBuffer.end(), offset); // name
    offset += counterNameBuffer.size();
    std::copy(counterDescriptionBuffer.begin(), counterDescriptionBuffer.end(), offset); // description
    if (includeUnits)
    {
        offset += counterDescriptionBuffer.size();
        std::copy(counterUnitsBuffer.begin(), counterUnitsBuffer.end(), offset); // units
    }
    ARMNN_NO_CONVERSION_WARN_END

    return true;
}

void SendCounterPacket::SendCounterDirectoryPacket(const ICounterDirectory& counterDirectory)
{
    // Get the amount of data that needs to be put into the packet
    const uint16_t categoryCount    = counterDirectory.GetCategoryCount();
    const uint16_t deviceCount      = counterDirectory.GetDeviceCount();
    const uint16_t counterSetCount  = counterDirectory.GetCounterSetCount();

    // Utils
    const size_t uint32_t_size = sizeof(uint32_t);
    const size_t packetHeaderSize = 2u;
    const size_t bodyHeaderSize = 6u;
    const uint32_t bodyHeaderSizeBytes = bodyHeaderSize * uint32_t_size;

    // Initialize the offset for the pointer tables
    uint32_t pointerTableOffset = 0;

    // --------------
    // Device records
    // --------------

    // Process device records
    std::vector<DeviceRecord> deviceRecords(deviceCount);
    const Devices& devices = counterDirectory.GetDevices();
    std::vector<uint32_t> deviceRecordOffsets(deviceCount, 0); // device_records_pointer_table
    size_t deviceRecordsSize = 0;
    size_t deviceIndex = 0;
    size_t deviceRecordOffsetIndex = 0;

    pointerTableOffset = armnn::numeric_cast<uint32_t>(deviceCount * uint32_t_size +
                                                       counterSetCount * uint32_t_size +
                                                       categoryCount   * uint32_t_size);
    for (auto it = devices.begin(); it != devices.end(); it++)
    {
        const DevicePtr& device = it->second;
        DeviceRecord& deviceRecord = deviceRecords.at(deviceIndex);

        std::string errorMessage;
        if (!CreateDeviceRecord(device, deviceRecord, errorMessage))
        {
            CancelOperationAndThrow<RuntimeException>(errorMessage);
        }

        // Update the total size in words of the device records
        deviceRecordsSize += deviceRecord.size();

        // Add the device record offset to the device records pointer table offset field
        deviceRecordOffsets[deviceRecordOffsetIndex] = pointerTableOffset;
        pointerTableOffset += armnn::numeric_cast<uint32_t>(deviceRecord.size() * uint32_t_size);

        deviceIndex++;
        deviceRecordOffsetIndex++;
    }

    // -------------------
    // Counter set records
    // -------------------

    // Process counter set records
    std::vector<CounterSetRecord> counterSetRecords(counterSetCount);
    const CounterSets& counterSets = counterDirectory.GetCounterSets();
    std::vector<uint32_t> counterSetRecordOffsets(counterSetCount, 0); // counter_set_records_pointer_table
    size_t counterSetRecordsSize = 0;
    size_t counterSetIndex = 0;
    size_t counterSetRecordOffsetIndex = 0;

    pointerTableOffset -= armnn::numeric_cast<uint32_t>(deviceCount * uint32_t_size);
    for (auto it = counterSets.begin(); it != counterSets.end(); it++)
    {
        const CounterSetPtr& counterSet = it->second;
        CounterSetRecord& counterSetRecord = counterSetRecords.at(counterSetIndex);

        std::string errorMessage;
        if (!CreateCounterSetRecord(counterSet, counterSetRecord, errorMessage))
        {
            CancelOperationAndThrow<RuntimeException>(errorMessage);
        }

        // Update the total size in words of the counter set records
        counterSetRecordsSize += counterSetRecord.size();

        // Add the counter set record offset to the counter set records pointer table offset field
        counterSetRecordOffsets[counterSetRecordOffsetIndex] = pointerTableOffset;
        pointerTableOffset += armnn::numeric_cast<uint32_t>(counterSetRecord.size() * uint32_t_size);

        counterSetIndex++;
        counterSetRecordOffsetIndex++;
    }

    // ----------------
    // Category records
    // ----------------

    // Process category records
    std::vector<CategoryRecord> categoryRecords(categoryCount);
    const Categories& categories = counterDirectory.GetCategories();
    std::vector<uint32_t> categoryRecordOffsets(categoryCount, 0); // category_records_pointer_table
    size_t categoryRecordsSize = 0;
    size_t categoryIndex = 0;
    size_t categoryRecordOffsetIndex = 0;

    pointerTableOffset -= armnn::numeric_cast<uint32_t>(counterSetCount * uint32_t_size);
    for (auto it = categories.begin(); it != categories.end(); it++)
    {
        const CategoryPtr& category = *it;
        CategoryRecord& categoryRecord = categoryRecords.at(categoryIndex);

        std::string errorMessage;
        if (!CreateCategoryRecord(category, counterDirectory.GetCounters(), categoryRecord, errorMessage))
        {
            CancelOperationAndThrow<RuntimeException>(errorMessage);
        }

        // Update the total size in words of the category records
        categoryRecordsSize += categoryRecord.size();

        // Add the category record offset to the category records pointer table offset field
        categoryRecordOffsets[categoryRecordOffsetIndex] = pointerTableOffset;
        pointerTableOffset += armnn::numeric_cast<uint32_t>(categoryRecord.size() * uint32_t_size);

        categoryIndex++;
        categoryRecordOffsetIndex++;
    }

    // Calculate the length in words of the counter directory packet's data (excludes the packet header size)
    const size_t counterDirectoryPacketDataLength =
                 bodyHeaderSize +                 // The size of the body header
                 deviceRecordOffsets.size() +     // The size of the device records pointer table
                 counterSetRecordOffsets.size() + // The size of counter set pointer table
                 categoryRecordOffsets.size() +   // The size of category records pointer table
                 deviceRecordsSize +              // The total size of the device records
                 counterSetRecordsSize +          // The total size of the counter set records
                 categoryRecordsSize;             // The total size of the category records

    // Calculate the size in words of the counter directory packet (the data length plus the packet header size)
    const size_t counterDirectoryPacketSize = packetHeaderSize +                // The size of the packet header
                                              counterDirectoryPacketDataLength; // The data length

    // Allocate the necessary space for the counter directory packet
    std::vector<uint32_t> counterDirectoryPacket(counterDirectoryPacketSize, 0);

    // -------------
    // Packet header
    // -------------

    // Packet header word 0:
    // 26:31 [6]  packet_family: control Packet Family
    // 16:25 [10] packet_id: packet identifier
    // 8:15  [8]  reserved: all zeros
    // 0:7   [8]  reserved: all zeros
    uint32_t packetFamily = 0;
    uint32_t packetId = 2;
    uint32_t packetHeaderWord0 = ((packetFamily & 0x3F) << 26) | ((packetId & 0x3FF) << 16);

    // Packet header word 1:
    // 0:31 [32] data_length: length of data, in bytes
    uint32_t packetHeaderWord1 = armnn::numeric_cast<uint32_t>(
            counterDirectoryPacketDataLength * uint32_t_size);

    // Create the packet header
    uint32_t packetHeader[2]
    {
        packetHeaderWord0, // packet_family + packet_id + reserved + reserved
        packetHeaderWord1  // data_length
    };

    // -----------
    // Body header
    // -----------

    // Body header word 0:
    // 16:31 [16] device_records_count: number of entries in the device_records_pointer_table
    // 0:15  [16] reserved: all zeros
    const uint32_t bodyHeaderWord0 = static_cast<uint32_t>(deviceCount) << 16;

    // Body header word 1:
    // 0:31 [32] device_records_pointer_table_offset: offset to the device_records_pointer_table
    const uint32_t bodyHeaderWord1 = bodyHeaderSizeBytes; // The offset is always the bodyHeaderSize,
                                                          // as the device record pointer table field
                                                          // is always the first item in the pool

    // Body header word 2:
    // 16:31 [16] counter_set_count: number of entries in the counter_set_pointer_table
    // 0:15  [16] reserved: all zeros
    const uint32_t bodyHeaderWord2 = static_cast<uint32_t>(counterSetCount) << 16;

    // Body header word 3:
    // 0:31 [32] counter_set_pointer_table_offset: offset to the counter_set_pointer_table
    const uint32_t bodyHeaderWord3 = armnn::numeric_cast<uint32_t>(deviceRecordOffsets.size() *
                                                                   uint32_t_size +       // The size of the
                                                                   bodyHeaderSizeBytes); // device records pointer table

    // Body header word 4:
    // 16:31 [16] categories_count: number of entries in the categories_pointer_table
    // 0:15  [16] reserved: all zeros
    const uint32_t bodyHeaderWord4 = static_cast<uint32_t>(categoryCount) << 16;

    // Body header word 3:
    // 0:31 [32] categories_pointer_table_offset: offset to the categories_pointer_table
    const uint32_t bodyHeaderWord5 =
                   armnn::numeric_cast<uint32_t>(
                       deviceRecordOffsets.size() * uint32_t_size +     // The size of the device records
                       counterSetRecordOffsets.size() * uint32_t_size   // pointer table, plus the size of
                       +  bodyHeaderSizeBytes);                         // the counter set pointer table

    // Create the body header
    const uint32_t bodyHeader[bodyHeaderSize]
    {
        bodyHeaderWord0, // device_records_count + reserved
        bodyHeaderWord1, // device_records_pointer_table_offset
        bodyHeaderWord2, // counter_set_count + reserved
        bodyHeaderWord3, // counter_set_pointer_table_offset
        bodyHeaderWord4, // categories_count + reserved
        bodyHeaderWord5  // categories_pointer_table_offset
    };

    ARMNN_NO_CONVERSION_WARN_BEGIN
    // Create the counter directory packet
    auto counterDirectoryPacketOffset = counterDirectoryPacket.begin();
    // packet_header
    std::copy(packetHeader, packetHeader + packetHeaderSize, counterDirectoryPacketOffset);
    counterDirectoryPacketOffset += packetHeaderSize;
    // body_header
    std::copy(bodyHeader, bodyHeader + bodyHeaderSize, counterDirectoryPacketOffset);
    counterDirectoryPacketOffset += bodyHeaderSize;
    // device_records_pointer_table
    std::copy(deviceRecordOffsets.begin(), deviceRecordOffsets.end(), counterDirectoryPacketOffset);
    counterDirectoryPacketOffset += deviceRecordOffsets.size();
    // counter_set_pointer_table
    std::copy(counterSetRecordOffsets.begin(), counterSetRecordOffsets.end(), counterDirectoryPacketOffset);
    counterDirectoryPacketOffset += counterSetRecordOffsets.size();
    // category_pointer_table
    std::copy(categoryRecordOffsets.begin(), categoryRecordOffsets.end(), counterDirectoryPacketOffset);
    counterDirectoryPacketOffset += categoryRecordOffsets.size();
    // device_records
    for (const DeviceRecord& deviceRecord : deviceRecords)
    {
        std::copy(deviceRecord.begin(), deviceRecord.end(), counterDirectoryPacketOffset); // device_record
        counterDirectoryPacketOffset += deviceRecord.size();
    }
    // counter_set_records
    for (const CounterSetRecord& counterSetRecord : counterSetRecords)
    {
        std::copy(counterSetRecord.begin(), counterSetRecord.end(), counterDirectoryPacketOffset); // counter_set_record
        counterDirectoryPacketOffset += counterSetRecord.size();
    }
    // category_records
    for (const CategoryRecord& categoryRecord : categoryRecords)
    {
        std::copy(categoryRecord.begin(), categoryRecord.end(), counterDirectoryPacketOffset); // category_record
        counterDirectoryPacketOffset += categoryRecord.size();
    }
    ARMNN_NO_CONVERSION_WARN_END

    // Calculate the total size in bytes of the counter directory packet
    uint32_t totalSize = armnn::numeric_cast<uint32_t>(counterDirectoryPacketSize * uint32_t_size);

    // Reserve space in the buffer for the packet
    uint32_t reserved = 0;
    IPacketBufferPtr writeBuffer = m_BufferManager.Reserve(totalSize, reserved);

    if (writeBuffer == nullptr || reserved < totalSize)
    {
        CancelOperationAndThrow<BufferExhaustion>(
            writeBuffer,
            fmt::format("No space left in buffer. Unable to reserve ({}) bytes.", totalSize));
    }

    // Offset for writing to the buffer
    uint32_t offset = 0;

    // Write the counter directory packet to the buffer
    for (uint32_t counterDirectoryPacketWord : counterDirectoryPacket)
    {
        WriteUint32(writeBuffer, offset, counterDirectoryPacketWord);
        offset += armnn::numeric_cast<uint32_t>(uint32_t_size);
    }

    m_BufferManager.Commit(writeBuffer, totalSize);
}

void SendCounterPacket::SendPeriodicCounterCapturePacket(uint64_t timestamp, const IndexValuePairsVector& values)
{
    uint32_t uint16_t_size = sizeof(uint16_t);
    uint32_t uint32_t_size = sizeof(uint32_t);
    uint32_t uint64_t_size = sizeof(uint64_t);

    uint32_t packetFamily = 3;
    uint32_t packetClass = 0;
    uint32_t packetType = 0;
    uint32_t headerSize = 2 * uint32_t_size;
    uint32_t bodySize = uint64_t_size + armnn::numeric_cast<uint32_t>(values.size()) * (uint16_t_size + uint32_t_size);
    uint32_t totalSize = headerSize + bodySize;
    uint32_t offset = 0;
    uint32_t reserved = 0;

    IPacketBufferPtr writeBuffer = m_BufferManager.Reserve(totalSize, reserved);

    if (writeBuffer == nullptr || reserved < totalSize)
    {
        CancelOperationAndThrow<BufferExhaustion>(
            writeBuffer,
            fmt::format("No space left in buffer. Unable to reserve ({}) bytes.", totalSize));
    }

    // Create header.
    WriteUint32(writeBuffer,
                offset,
                ((packetFamily & 0x0000003F) << 26) |
                ((packetClass  & 0x0000007F) << 19) |
                ((packetType   & 0x00000007) << 16));
    offset += uint32_t_size;
    WriteUint32(writeBuffer, offset, bodySize);

    // Copy captured Timestamp.
    offset += uint32_t_size;
    WriteUint64(writeBuffer, offset, timestamp);

    // Copy selectedCounterIds.
    offset += uint64_t_size;
    for (const auto& pair: values)
    {
        WriteUint16(writeBuffer, offset, pair.counterId);
        offset += uint16_t_size;
        WriteUint32(writeBuffer, offset, pair.counterValue);
        offset += uint32_t_size;
    }

    m_BufferManager.Commit(writeBuffer, totalSize);
}

void SendCounterPacket::SendPeriodicCounterSelectionPacket(uint32_t capturePeriod,
                                                           const std::vector<uint16_t>& selectedCounterIds)
{
    uint32_t uint16_t_size = sizeof(uint16_t);
    uint32_t uint32_t_size = sizeof(uint32_t);

    uint32_t packetFamily = 0;
    uint32_t packetId = 4;
    uint32_t headerSize = 2 * uint32_t_size;
    uint32_t bodySize = uint32_t_size + armnn::numeric_cast<uint32_t>(selectedCounterIds.size()) * uint16_t_size;
    uint32_t totalSize = headerSize + bodySize;
    uint32_t offset = 0;
    uint32_t reserved = 0;

    IPacketBufferPtr writeBuffer = m_BufferManager.Reserve(totalSize, reserved);

    if (writeBuffer == nullptr || reserved < totalSize)
    {
        CancelOperationAndThrow<BufferExhaustion>(
            writeBuffer,
            fmt::format("No space left in buffer. Unable to reserve ({}) bytes.", totalSize));
    }

    // Create header.
    WriteUint32(writeBuffer, offset, ((packetFamily & 0x3F) << 26) | ((packetId & 0x3FF) << 16));
    offset += uint32_t_size;
    WriteUint32(writeBuffer, offset, bodySize);

    // Copy capturePeriod.
    offset += uint32_t_size;
    WriteUint32(writeBuffer, offset, capturePeriod);

    // Copy selectedCounterIds.
    offset += uint32_t_size;
    for(const uint16_t& id: selectedCounterIds)
    {
        WriteUint16(writeBuffer, offset, id);
        offset += uint16_t_size;
    }

    m_BufferManager.Commit(writeBuffer, totalSize);
}

} // namespace profiling

} // namespace armnn
