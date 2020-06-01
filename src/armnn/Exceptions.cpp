//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <armnn/Exceptions.hpp>

#include <string>

namespace armnn
{

Exception::Exception(const std::string& message)
: m_Message{message}
{
}

Exception::Exception(const std::string& message,
                     const CheckLocation& location)
: m_Message{message}
{
    m_Message += location.AsString();
}

Exception::Exception(const Exception& other,
                     const std::string& message,
                     const CheckLocation& location)
: m_Message{other.m_Message}
{
    m_Message += "\n" + message + location.AsString();
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
