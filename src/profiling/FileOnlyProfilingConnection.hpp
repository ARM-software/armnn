//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/profiling/ILocalPacketHandler.hpp>
#include "DirectoryCaptureCommandHandler.hpp"
#include "IProfilingConnection.hpp"
#include <Packet.hpp>
#include "ProfilingUtils.hpp"
#include "Runtime.hpp"

#include <atomic>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <queue>
#include <thread>

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
        , m_IsRunning(false)
        , m_KeepRunning(false)
        , m_Timeout(1000)
    {
        for (ILocalPacketHandlerSharedPtr localPacketHandler : options.m_LocalPacketHandlers)
        {
            AddLocalPacketHandler(localPacketHandler);
        }
        if (!options.m_LocalPacketHandlers.empty())
        {
            StartProcessingThread();
        }
        // NOTE: could add timeout to the external profiling options
    };

    ~FileOnlyProfilingConnection();

    bool IsOpen() const override;

    void Close() override;

    // This is effectively receiving a data packet from ArmNN.
    bool WritePacket(const unsigned char* buffer, uint32_t length) override;

    // Sending a packet back to ArmNN.
    Packet ReadPacket(uint32_t timeout) override;

private:
    void AddLocalPacketHandler(ILocalPacketHandlerSharedPtr localPacketHandler);
    void StartProcessingThread();
    void ClearReadableList();
    void DispatchPacketToHandlers(const Packet& packet);

    bool WaitForStreamMeta(const unsigned char* buffer, uint32_t length);

    uint32_t ToUint32(const unsigned char* data, TargetEndianness endianness);

    void SendConnectionAck();

    bool SendCounterSelectionPacket();

    PackageActivity GetPackageActivity(const Packet& packet, uint32_t headerAsWords[2]);

    void Fail(const std::string& errorMessage);

    void ForwardPacketToHandlers(Packet& packet);
    void ServiceLocalHandlers();

    Runtime::CreationOptions::ExternalProfilingOptions m_Options;
    bool m_QuietOp;
    std::vector<uint16_t> m_IdList;
    std::queue<Packet> m_PacketQueue;
    TargetEndianness m_Endianness;

    std::mutex m_PacketAvailableMutex;
    std::condition_variable m_ConditionPacketAvailable;

    std::vector<ILocalPacketHandlerSharedPtr> m_PacketHandlers;
    std::map<uint32_t, std::vector<ILocalPacketHandlerSharedPtr>> m_IndexedHandlers;
    std::vector<ILocalPacketHandlerSharedPtr> m_UniversalHandlers;

    // List of readable packets for the local packet handlers
    std::queue<Packet> m_ReadableList;
    // Mutex and condition variable for the readable packet list
    std::mutex m_ReadableMutex;
    std::condition_variable m_ConditionPacketReadable;
    // thread that takes items from the readable list and dispatches them
    // to the handlers.
    std::thread m_LocalHandlersThread;
    // atomic booleans that control the operation of the local handlers thread
    std::atomic<bool> m_IsRunning;
    std::atomic<bool> m_KeepRunning;
    int m_Timeout;
};

}    // namespace profiling

}    // namespace armnn