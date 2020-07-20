//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

#include "NetworkSockets.hpp"

namespace arm
{

namespace pipe
{

/// Socket Connection Exception for profiling
class SocketConnectionException : public std::exception
{
public:
    explicit SocketConnectionException(const std::string &message, arm::pipe::Socket socket)
        : m_Message(message), m_Socket(socket), m_ErrNo(-1) {};

    explicit SocketConnectionException(const std::string &message, arm::pipe::Socket socket, int errNo)
        : m_Message(message), m_Socket(socket), m_ErrNo(errNo) {};

    /// @return - Error message of  SocketProfilingConnection
    virtual const char *what() const noexcept override
    {
        return m_Message.c_str();
    }

    /// @return - Socket File Descriptor of SocketProfilingConnection
    ///           or '-1', an invalid file descriptor
    arm::pipe::Socket GetSocketFd() const noexcept
    {
        return m_Socket;
    }

    /// @return - errno of SocketProfilingConnection
    int GetErrorNo() const noexcept
    {
        return m_ErrNo;
    }

private:
    std::string m_Message;
    arm::pipe::Socket m_Socket;
    int m_ErrNo;
};
} // namespace pipe
} // namespace arm
