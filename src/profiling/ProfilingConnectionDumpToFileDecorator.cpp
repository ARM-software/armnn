//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingConnectionDumpToFileDecorator.hpp"

#include <common/include/NumericCast.hpp>
#include <common/include/ProfilingException.hpp>

#include <fstream>

namespace arm
{

namespace pipe
{

ProfilingConnectionDumpToFileDecorator::ProfilingConnectionDumpToFileDecorator(
    std::unique_ptr<IProfilingConnection> connection,
    const ProfilingOptions& options,
    bool ignoreFailures)
      : m_Connection(std::move(connection))
      , m_Options(options)
      , m_IgnoreFileErrors(ignoreFailures)
{
    if (!m_Connection)
    {
        throw arm::pipe::InvalidArgumentException("Connection cannot be nullptr");
    }
}

ProfilingConnectionDumpToFileDecorator::~ProfilingConnectionDumpToFileDecorator()
{
    Close();
}

bool ProfilingConnectionDumpToFileDecorator::IsOpen() const
{
    return m_Connection->IsOpen();
}

void ProfilingConnectionDumpToFileDecorator::Close()
{
    m_IncomingDumpFileStream.flush();
    m_IncomingDumpFileStream.close();
    m_OutgoingDumpFileStream.flush();
    m_OutgoingDumpFileStream.close();
    m_Connection->Close();
}

bool ProfilingConnectionDumpToFileDecorator::WritePacket(const unsigned char* buffer, uint32_t length)
{
    bool success = true;
    if (!m_Options.m_OutgoingCaptureFile.empty())
    {
        success &= DumpOutgoingToFile(buffer, length);
    }
    success &= m_Connection->WritePacket(buffer, length);
    return success;
}

arm::pipe::Packet ProfilingConnectionDumpToFileDecorator::ReadPacket(uint32_t timeout)
{
    arm::pipe::Packet packet = m_Connection->ReadPacket(timeout);
    if (!m_Options.m_IncomingCaptureFile.empty())
    {
        DumpIncomingToFile(packet);
    }
    return packet;
}

bool ProfilingConnectionDumpToFileDecorator::OpenIncomingDumpFile()
{
    m_IncomingDumpFileStream.open(m_Options.m_IncomingCaptureFile, std::ios::out | std::ios::binary);
    return m_IncomingDumpFileStream.is_open();
}

bool ProfilingConnectionDumpToFileDecorator::OpenOutgoingDumpFile()
{
    m_OutgoingDumpFileStream.open(m_Options.m_OutgoingCaptureFile, std::ios::out | std::ios::binary);
    return m_OutgoingDumpFileStream.is_open();
}


/// Dumps incoming data into the file specified by m_Settings.m_IncomingDumpFileName.
/// If m_IgnoreFileErrors is set to true in m_Settings, write errors will be ignored,
/// i.e. the method will not throw an exception if it encounters an error while trying
/// to write the data into the specified file.
/// @param packet data packet to write
/// @return nothing
void ProfilingConnectionDumpToFileDecorator::DumpIncomingToFile(const arm::pipe::Packet& packet)
{
    bool success = true;
    if (!m_IncomingDumpFileStream.is_open())
    {
        // attempt to open dump file
        success &= OpenIncomingDumpFile();
        if (!(success || m_IgnoreFileErrors))
        {
            Fail("Failed to open \"" + m_Options.m_IncomingCaptureFile + "\" for writing");
        }
    }

    // attempt to write binary data from packet
    const unsigned int header       = packet.GetHeader();
    const unsigned int packetLength = packet.GetLength();

    m_IncomingDumpFileStream.write(reinterpret_cast<const char*>(&header), sizeof header);
    m_IncomingDumpFileStream.write(reinterpret_cast<const char*>(&packetLength), sizeof packetLength);
    m_IncomingDumpFileStream.write(reinterpret_cast<const char*>(packet.GetData()),
                                   arm::pipe::numeric_cast<std::streamsize>(packetLength));

    success &= m_IncomingDumpFileStream.good();
    if (!(success || m_IgnoreFileErrors))
    {
        Fail("Error writing incoming packet of " + std::to_string(packetLength) + " bytes");
    }
}

/// Dumps outgoing data into the file specified by m_Settings.m_OutgoingDumpFileName.
/// If m_IgnoreFileErrors is set to true in m_Settings, write errors will be ignored,
/// i.e. the method will not throw an exception if it encounters an error while trying
/// to write the data into the specified file. However, the return value will still
/// signal if the write has not been completed succesfully.
/// @param buffer pointer to data to write
/// @param length number of bytes to write
/// @return true if write successful, false otherwise
bool ProfilingConnectionDumpToFileDecorator::DumpOutgoingToFile(const unsigned char* buffer, uint32_t length)
{
    bool success = true;
    if (!m_OutgoingDumpFileStream.is_open())
    {
        // attempt to open dump file
        success &= OpenOutgoingDumpFile();
        if (!(success || m_IgnoreFileErrors))
        {
            Fail("Failed to open \"" + m_Options.m_OutgoingCaptureFile + "\" for writing");
        }
    }

    // attempt to write binary data
    m_OutgoingDumpFileStream.write(reinterpret_cast<const char*>(buffer),
                                   arm::pipe::numeric_cast<std::streamsize>(length));
    success &= m_OutgoingDumpFileStream.good();
    if (!(success || m_IgnoreFileErrors))
    {
        Fail("Error writing outgoing packet of " + std::to_string(length) + " bytes");
    }

    return success;
}

void ProfilingConnectionDumpToFileDecorator::Fail(const std::string& errorMessage)
{
    Close();
    throw arm::pipe::ProfilingException(errorMessage);
}

} // namespace pipe

} // namespace arm
