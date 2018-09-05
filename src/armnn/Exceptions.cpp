//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "armnn/Exceptions.hpp"

#include <string>

namespace armnn
{

Exception::Exception(const std::string& message)
: m_Message(message)
{
}

const char* Exception::what() const noexcept
{
    return m_Message.c_str();
}

UnimplementedException::UnimplementedException()
: Exception("Function not yet implemented")
{
}

}
