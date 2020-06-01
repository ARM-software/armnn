//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <string>

namespace armnn
{
class TensorShape;
} // namespace armnn

namespace armnnUtils
{

/// Converts an int value into the Prototxt octal representation
std::string ConvertInt32ToOctalString(int value);

/// Converts an TensorShape into Prototxt representation
std::string ConvertTensorShapeToString(const armnn::TensorShape& shape);

} // namespace armnnUtils
