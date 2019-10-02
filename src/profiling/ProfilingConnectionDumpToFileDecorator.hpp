//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IProfilingConnection.hpp"

#include <armnn/Optional.hpp>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace armnn
{

namespace profiling
{

class ProfilingConnectionDumpToFileDecorator : public IProfilingConnection
{
public:
    struct Settings
    {
        Settings(const std::string& incomingDumpFileName = "",
                 const std::string& outgoingDumpFileName = "",
                 bool ignoreFileErrors = true)
            : m_IncomingDumpFileName(incomingDumpFileName)
            , m_OutgoingDumpFileName(outgoingDumpFileName)
            , m_DumpIncoming(!incomingDumpFileName.empty())
            , m_DumpOutgoing(!outgoingDumpFileName.empty())
            , m_IgnoreFileErrors(ignoreFileErrors)
        {}

        ~Settings() = default;

        std::string m_IncomingDumpFileName;
        std::string m_OutgoingDumpFileName;
        bool        m_DumpIncoming;
        bool        m_DumpOutgoing;
        bool        m_IgnoreFileErrors;
    };

    ProfilingConnectionDumpToFileDecorator(std::unique_ptr<IProfilingConnection> connection,
                                           const Settings& settings);

    ~ProfilingConnectionDumpToFileDecorator();

    bool IsOpen() const override;

    void Close() override;

    bool WritePacket(const unsigned char* buffer, uint32_t length) override;

    Packet ReadPacket(uint32_t timeout) override;

private:
    bool OpenIncomingDumpFile();

    bool OpenOutgoingDumpFile();

    void DumpIncomingToFile(const Packet& packet);

    bool DumpOutgoingToFile(const char* buffer, uint32_t length);

    void Fail(const std::string& errorMessage);

    std::unique_ptr<IProfilingConnection> m_Connection;
    Settings                              m_Settings;
    std::ofstream                         m_IncomingDumpFileStream;
    std::ofstream                         m_OutgoingDumpFileStream;
};

using ProfilingConnectionDumpToFileDecoratorSettings = ProfilingConnectionDumpToFileDecorator::Settings;

} // namespace profiling

} // namespace armnn
