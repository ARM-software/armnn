//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PrototxtConversions.hpp"

#include <boost/format.hpp>

#include <iomanip>
#include <sstream>
#include <string>

namespace armnnUtils
{

/// Converts an int value into the Prototxt octal representation
std::string ConvertInt32ToOctalString(int value)
{
    std::stringstream ss;
    std::string returnString;
    for (int i = 0; i < 4; ++i)
    {
        ss << "\\";
        ss << std::setw(3) << std::setfill('0') << std::oct << ((value >> (i * 8)) & 0xFF);
    }

    ss >> returnString;
    return returnString;
}

} // namespace armnnUtils
