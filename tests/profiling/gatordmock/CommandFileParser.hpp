//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "GatordMockService.hpp"
#include <string>

namespace armnn
{

namespace gatordmock
{

/// This class parses a command file for the GatordMockService. The file contains one command per line.
/// Valid commands are: SET and WAIT.
///
///  SET: Will construct and send a PeriodicCounterSelection packet to enable a set of counters.
///  WAIT: Will pause for a set period of time to allow for data to be received.
class CommandFileParser
{
public:
    void ParseFile(std::string CommandFile, GatordMockService& mockService);
};

}    // namespace gatordmock
}    // namespace armnn
