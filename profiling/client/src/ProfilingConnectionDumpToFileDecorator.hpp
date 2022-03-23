//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IProfilingConnection.hpp"
#include "ProfilingUtils.hpp"

#include <client/include/ProfilingOptions.hpp>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace arm
{

namespace pipe
{

class ProfilingConnectionDumpToFileDecorator : public IProfilingConnection
{
public:

    ProfilingConnectionDumpToFileDecorator(std::unique_ptr<IProfilingConnection> connection,
                                           const ProfilingOptions& options,
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

    std::unique_ptr<IProfilingConnection> m_Connection;
    ProfilingOptions                      m_Options;
    std::ofstream                         m_IncomingDumpFileStream;
    std::ofstream                         m_OutgoingDumpFileStream;
    bool                                  m_IgnoreFileErrors;
};

} // namespace pipe

} // namespace arm
