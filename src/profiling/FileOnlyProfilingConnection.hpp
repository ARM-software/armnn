//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/profiling/ILocalPacketHandler.hpp>
#include "DirectoryCaptureCommandHandler.hpp"
#include "IProfilingConnection.hpp"
#include "ProfilingUtils.hpp"
#include "Runtime.hpp"

#include <common/include/Packet.hpp>

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

// forward declaration
class FileOnlyProfilingConnection;

class StreamMetaDataProcessor : public ILocalPacketHandler
{
public:
    explicit StreamMetaDataProcessor(FileOnlyProfilingConnection* fileOnlyProfilingConnection) :
            m_FileOnlyProfilingConnection(fileOnlyProfilingConnection),
            m_MetaDataPacketHeader(ConstructHeader(0, 0)) {};

    std::vector<uint32_t> GetHeadersAccepted() override;

    void HandlePacket(const arm::pipe::Packet& packet) override;

private:
    FileOnlyProfilingConnection* m_FileOnlyProfilingConnection;
    uint32_t m_MetaDataPacketHeader;

    static uint32_t ToUint32(const unsigned char* data, TargetEndianness endianness);
};

class FileOnlyProfilingConnection : public IProfilingConnection, public IInternalProfilingConnection
{
public:
    explicit FileOnlyProfilingConnection(const IRuntime::CreationOptions::ExternalProfilingOptions& options)
        : m_Options(options)
        , m_Endianness(TargetEndianness::LeWire)    // Set a sensible default.
                                                    // StreamMetaDataProcessor will set a real value.
        , m_IsRunning(false)
        , m_KeepRunning(false)
        , m_Timeout(1000)
    {
        // add the StreamMetaDataProcessor
        auto streamMetaDataProcessor = std::make_shared<StreamMetaDataProcessor>(this);
        AddLocalPacketHandler(streamMetaDataProcessor);
        // and any additional ones added by the users
        for (const ILocalPacketHandlerSharedPtr& localPacketHandler : options.m_LocalPacketHandlers)
        {
            AddLocalPacketHandler(localPacketHandler);
        }
        if (!m_PacketHandlers.empty())
        {
            StartProcessingThread();
        }
        // NOTE: could add timeout to the external profiling options
    };

    ~FileOnlyProfilingConnection() override;

    bool IsOpen() const override;

    void Close() override;

    // This is effectively receiving a data packet from ArmNN.
    bool WritePacket(const unsigned char* buffer, uint32_t length) override;

    // Sending a packet back to ArmNN.
    arm::pipe::Packet ReadPacket(uint32_t timeout) override;

    void SetEndianess(const TargetEndianness& endianness) override //IInternalProfilingConnection
    {
        m_Endianness = endianness;
    }

    void ReturnPacket(arm::pipe::Packet& packet) override; //IInternalProfilingConnection

private:
    void AddLocalPacketHandler(ILocalPacketHandlerSharedPtr localPacketHandler);
    void StartProcessingThread();
    void ClearReadableList();
    void DispatchPacketToHandlers(const arm::pipe::Packet& packet);

    void Fail(const std::string& errorMessage);

    void ForwardPacketToHandlers(arm::pipe::Packet& packet);
    void ServiceLocalHandlers();

    IRuntime::CreationOptions::ExternalProfilingOptions m_Options;
    std::queue<arm::pipe::Packet> m_PacketQueue;
    TargetEndianness m_Endianness;

    std::mutex m_PacketAvailableMutex;
    std::condition_variable m_ConditionPacketAvailable;

    std::vector<ILocalPacketHandlerSharedPtr> m_PacketHandlers;
    std::map<uint32_t, std::vector<ILocalPacketHandlerSharedPtr>> m_IndexedHandlers;
    std::vector<ILocalPacketHandlerSharedPtr> m_UniversalHandlers;

    // List of readable packets for the local packet handlers
    std::queue<arm::pipe::Packet> m_ReadableList;
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