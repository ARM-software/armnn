//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingConnectionFactory.hpp"

#include "FileOnlyProfilingConnection.hpp"
#include "ProfilingConnectionDumpToFileDecorator.hpp"
#include "SocketProfilingConnection.hpp"

namespace arm
{

namespace pipe
{

std::unique_ptr<IProfilingConnection> ProfilingConnectionFactory::GetProfilingConnection(
    const ProfilingOptions& options) const
{
    // Before proceed to create the IProfilingConnection, check if the file format is supported
    if (!(options.m_FileFormat == "binary"))
    {
        throw arm::pipe::UnimplementedException("Unsupported profiling file format, only binary is supported");
    }

    // We can create 3 different types of IProfilingConnection.
    // 1: If no relevant options are specified then a SocketProfilingConnection is returned.
    // 2: If both incoming and outgoing capture files are specified then a SocketProfilingConnection decorated by a
    //    ProfilingConnectionDumpToFileDecorator is returned.
    // 3: If both incoming and outgoing capture files are specified and "file only" then a FileOnlyProfilingConnection
    //    decorated by a ProfilingConnectionDumpToFileDecorator is returned.
    // 4. There is now another option if m_FileOnly == true and there are ILocalPacketHandlers specified
    //    we can create a FileOnlyProfilingConnection without a file dump
    if ((!options.m_IncomingCaptureFile.empty() || !options.m_OutgoingCaptureFile.empty()) && !options.m_FileOnly)
    {
        // This is type 2.
        return std::make_unique<ProfilingConnectionDumpToFileDecorator>(std::make_unique<SocketProfilingConnection>(),
                                                                        options);
    }
    else if ((!options.m_IncomingCaptureFile.empty() || !options.m_OutgoingCaptureFile.empty()) && options.m_FileOnly)
    {
        // This is type 3.
        return std::make_unique<ProfilingConnectionDumpToFileDecorator>(
            std::make_unique<FileOnlyProfilingConnection>(options), options);
    }
    else if (options.m_FileOnly && !options.m_LocalPacketHandlers.empty())
    {
        // This is the type 4.
        return std::make_unique<FileOnlyProfilingConnection>(options);
    }
    else
    {
        // This is type 1.
        return std::make_unique<SocketProfilingConnection>();
    }
}

}    // namespace pipe

}    // namespace arm
