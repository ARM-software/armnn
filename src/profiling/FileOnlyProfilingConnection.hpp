//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CounterDirectory.hpp"
#include "DirectoryCaptureCommandHandler.hpp"
#include "IProfilingConnection.hpp"
#include "ProfilingUtils.hpp"
#include "Runtime.hpp"

#include <condition_variable>
#include <fstream>
#include <queue>

namespace armnn
{

namespace profiling
{

enum class TargetEndianness
{
    BeWire,
    LeWire
};

enum class PackageActivity
{
    StreamMetaData,
    CounterDirectory,
    Unknown
};

class FileOnlyProfilingConnection : public IProfilingConnection
{
public:
    FileOnlyProfilingConnection(const Runtime::CreationOptions::ExternalProfilingOptions& options,
                                const bool quietOp = true)
        : m_Options(options)
        , m_QuietOp(quietOp)
        , m_Endianness(TargetEndianness::LeWire)    // Set a sensible default. WaitForStreamMeta will set a real value.
        {};

    ~FileOnlyProfilingConnection();

    bool IsOpen() const override;

    void Close() override;

    // This is effectively receiving a data packet from ArmNN.
    bool WritePacket(const unsigned char* buffer, uint32_t length) override;

    // Sending a packet back to ArmNN.
    Packet ReadPacket(uint32_t timeout) override;

private:
    bool WaitForStreamMeta(const unsigned char* buffer, uint32_t length);

    uint32_t ToUint32(const unsigned char* data, TargetEndianness endianness);

    void SendConnectionAck();

    bool SendCounterSelectionPacket();

    PackageActivity GetPackageActivity(const unsigned char* buffer, uint32_t headerAsWords[2]);

    void Fail(const std::string& errorMessage);

    static const uint32_t PIPE_MAGIC = 0x45495434;

    Runtime::CreationOptions::ExternalProfilingOptions m_Options;
    bool m_QuietOp;
    std::vector<uint16_t> m_IdList;
    std::queue<Packet> m_PacketQueue;
    TargetEndianness m_Endianness;

    std::mutex m_PacketAvailableMutex;
    std::condition_variable m_ConditionPacketAvailable;
};

}    // namespace profiling

}    // namespace armnn