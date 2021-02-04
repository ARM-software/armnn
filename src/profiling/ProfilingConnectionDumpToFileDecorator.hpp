//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IProfilingConnection.hpp"
#include "ProfilingUtils.hpp"

#include <Runtime.hpp>
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

    ProfilingConnectionDumpToFileDecorator(std::unique_ptr<IProfilingConnection> connection,
                                           const IRuntime::CreationOptions::ExternalProfilingOptions& options,
                                           bool ignoreFailures = false);

    ~ProfilingConnectionDumpToFileDecorator();

    bool IsOpen() const override;

    void Close() override;

    bool WritePacket(const unsigned char* buffer, uint32_t length) override;

    arm::pipe::Packet ReadPacket(uint32_t timeout) override;

private:
    bool OpenIncomingDumpFile();

    bool OpenOutgoingDumpFile();

    void DumpIncomingToFile(const arm::pipe::Packet& packet);

    bool DumpOutgoingToFile(const unsigned char* buffer, uint32_t length);

    void Fail(const std::string& errorMessage);

    std::unique_ptr<IProfilingConnection>               m_Connection;
    IRuntime::CreationOptions::ExternalProfilingOptions m_Options;
    std::ofstream                                       m_IncomingDumpFileStream;
    std::ofstream                                       m_OutgoingDumpFileStream;
    bool                                                m_IgnoreFileErrors;
};

} // namespace profiling

} // namespace armnn
