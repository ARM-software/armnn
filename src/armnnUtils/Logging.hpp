//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once


#include "armnn/Utils.hpp"

#include <boost/log/trivial.hpp>

namespace armnnUtils
{

// Configures logging for the given Boost Log Core object.
void ConfigureLogging(boost::log::core* core,
                      bool              printToStandardOutput,
                      bool              printToDebugOutput,
                      armnn::LogSeverity severity);

}