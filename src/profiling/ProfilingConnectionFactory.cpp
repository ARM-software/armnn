//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingConnectionFactory.hpp"

#include "FileOnlyProfilingConnection.hpp"
#include "ProfilingConnectionDumpToFileDecorator.hpp"
#include "SocketProfilingConnection.hpp"

namespace armnn
{

namespace profiling
{

std::unique_ptr<IProfilingConnection> ProfilingConnectionFactory::GetProfilingConnection(
    const Runtime::CreationOptions::ExternalProfilingOptions& options) const
{
    // We can create 3 different types of IProfilingConnection.
    // 1: If no relevant options are specified then a SocketProfilingConnection is returned.
    // 2: If both incoming and outgoing capture files are specified then a SocketProfilingConnection decorated by a
    //    ProfilingConnectionDumpToFileDecorator is returned.
    // 3: If both incoming and outgoing capture files are specified and "file only" then a FileOnlyProfilingConnection
    //    decorated by a ProfilingConnectionDumpToFileDecorator is returned.
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
    else
    {
        // This is type 1.
        return std::make_unique<SocketProfilingConnection>();
    }
}

}    // namespace profiling

}    // namespace armnn
