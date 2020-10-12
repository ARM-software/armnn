//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PrototxtConversions.hpp"
#include "armnn/Tensor.hpp"

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

/// Converts an TensorShape into Prototxt representation
std::string ConvertTensorShapeToString(const armnn::TensorShape& shape)
{
    std::stringstream ss;
    for (unsigned int i = 0 ; i < shape.GetNumDimensions() ; i++)
    {
        ss << "dim {\n";
        ss << "size: " << std::to_string(shape[i]) << "\n";
        ss << "}\n";
    }
    return ss.str();

}
} // namespace armnnUtils
