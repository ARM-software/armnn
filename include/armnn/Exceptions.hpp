//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace armnn
{

struct CheckLocation
{
    const char* m_Function;
    const char* m_File;
    unsigned int m_Line;

    CheckLocation(const char* func,
                  const char* file,
                  unsigned int line)
    : m_Function{func}
    , m_File{file}
    , m_Line{line}
    {
    }

    std::string AsString() const
    {
        std::stringstream ss;
        ss << " at function " << m_Function
           << " [" << m_File << ':' << m_Line << "]";
        return ss.str();
    }

    std::string FileLine() const
    {
        std::stringstream ss;
        ss << " [" << m_File << ':' << m_Line << "]";
        return ss.str();
    }
};

/// Base class for all ArmNN exceptions so that users can filter to just those.
class Exception : public std::exception
{
public:
    explicit Exception(const std::string& message);

    /// exception with context
    explicit Exception(const std::string& message,
                       const CheckLocation& location);

    /// preserving previous exception context
    /// and adding local context information
    explicit Exception(const Exception& other,
                       const std::string& message,
                       const CheckLocation& location);

    virtual const char* what() const noexcept override;

private:
    std::string m_Message;
};

/// Class for non-fatal exceptions raised while initialising a backend
class BackendUnavailableException : public Exception
{
public:
    using Exception::Exception;
};

class ClRuntimeUnavailableException : public BackendUnavailableException
{
public:
    using BackendUnavailableException::BackendUnavailableException;
};

class InvalidArgumentException : public Exception
{
public:
    using Exception::Exception;
};

class FileNotFoundException : public Exception
{
public:
    using Exception::Exception;
};

class ParseException : public Exception
{
public:
    using Exception::Exception;
};

class UnimplementedException : public Exception
{
public:
    using Exception::Exception;
    UnimplementedException();
};

class LayerValidationException : public Exception
{
    using Exception::Exception;
};

class GraphValidationException : public Exception
{
    using Exception::Exception;
};

class BadOptionalAccessException : public Exception
{
    using Exception::Exception;
};

class RuntimeException : public Exception
{
    using Exception::Exception;
};

class MemoryImportException : public Exception
{
    using Exception::Exception;
};

class MemoryExportException : public Exception
{
    using Exception::Exception;
};

class TimeoutException : public Exception
{
    using Exception::Exception;
};

class BackendProfilingException : public Exception
{
public:
    using Exception::Exception;
};

class PolymorphicDowncastException : public Exception
{
public:
    using Exception::Exception;
};

class NullPointerException : public Exception
{
public:
    using Exception::Exception;
};

template <typename ExceptionType>
void ConditionalThrow(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw ExceptionType(message);
    }
}

template <typename ExceptionType>
void ConditionalThrow(bool condition)
{
    if (!condition)
    {
        throw ExceptionType();
    }
}


///
/// ComparedType must support:
///   operator==(const ComparedType&)
///   operator<<(ostream&, const ComparedType&)
///
template <typename ExceptionType, typename ComparedType>
void ConditionalThrowIfNotEqual(const std::string& message,
                                const ComparedType& leftHandSide,
                                const ComparedType& rightHandSide)
{
    if (!(leftHandSide == rightHandSide))
    {
        std::stringstream ss;
        ss << message << " : " << leftHandSide << " != " << rightHandSide;
        throw ExceptionType(ss.str());
    }
}

} // namespace armnn

#define CHECK_LOCATION() armnn::CheckLocation(__func__, __FILE__, __LINE__)
