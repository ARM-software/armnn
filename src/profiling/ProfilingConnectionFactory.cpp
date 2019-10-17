//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingConnectionFactory.hpp"
#include "SocketProfilingConnection.hpp"
#include "ProfilingConnectionDumpToFileDecorator.hpp"

namespace armnn
{

namespace profiling
{

std::unique_ptr<IProfilingConnection> ProfilingConnectionFactory::GetProfilingConnection(
    const Runtime::CreationOptions::ExternalProfilingOptions& options) const
{
    if ( !options.m_IncomingCaptureFile.empty() || !options.m_OutgoingCaptureFile.empty() )
    {
        bool ignoreFailures = false;
        return std::make_unique<ProfilingConnectionDumpToFileDecorator>(std::make_unique<SocketProfilingConnection>(),
                                                                        options,
                                                                        ignoreFailures);
    }
    else
    {
        return std::make_unique<SocketProfilingConnection>();
    }
}

} // namespace profiling

} // namespace armnn
