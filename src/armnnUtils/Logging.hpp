//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
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