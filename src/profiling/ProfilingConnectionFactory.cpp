//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingConnectionFactory.hpp"
#include "SocketProfilingConnection.hpp"

namespace armnn
{

namespace profiling
{

std::unique_ptr<IProfilingConnection> ProfilingConnectionFactory::GetProfilingConnection(
    const Runtime::CreationOptions::ExternalProfilingOptions& options) const
{
    return std::make_unique<SocketProfilingConnection>();
}

} // namespace profiling

} // namespace armnn
