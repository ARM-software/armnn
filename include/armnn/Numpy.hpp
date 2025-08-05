//
// Copyright © 2024-2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Copyright © 2021 Leon Merten Lohse
// SPDX-License-Identifier: MIT
//

#ifndef NUMPY_HPP
#define NUMPY_HPP

#include <fmt/format.h>
#include "Types.hpp"
#include "Tensor.hpp"

#include <fstream>
#include <iostream>

namespace armnnNumpy
{

    /// @struct HeaderInfo: contains information from the numpy file to be parsed.
    /// @var m_MajorVersion: Single byte containing the major version of numpy implementation
    /// @var m_MinorVersion: Single byte containing the minor version of numpy implementation
    /// @var m_MagicStringLength: unsigned 8bit int containing the length of the magic string.
    /// @var m_MagicString: Char array containing the magic string at the beginning of every numpy file.
    /// @var m_HeaderLen: 2 or 4byte unsigned int containing the length of the header, depending on version.
    struct HeaderInfo
    {
        char m_MajorVersion;
        char m_MinorVersion;
        const uint8_t m_MagicStringLength = 6;
        const char m_MagicString[7] = "\x93NUMPY";
        char m_HeaderLenBytes[4];
        uint32_t m_HeaderLen;
    };

    /// @struct Header: contains information from the numpy file to be parsed.
    /// @var m_HeaderString: String containing header.
    /// @var m_DescrString: String containing description.
    /// @var m_FortranOrder: Boolean declaring if array is in Fortran Order.
    /// @var m_Shape: Shape of the data.
    struct Header
    {
        std::string m_HeaderString;
        std::string m_DescrString;
        bool m_FortranOrder;
        std::vector<uint32_t> m_Shape;
    };

    inline void CreateHeaderInfo(std::ifstream &ifStream, HeaderInfo &headerInfo)
    {
        // A Numpy header consists of:
        // a magic string "x93NUMPY"
        // 1 byte for the major version
        // 1 byte for the minor version
        // 2 or 4 bytes for the header length
        // More info: https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
        char buffer[headerInfo.m_MagicStringLength + 2lu];
        ifStream.read(buffer, headerInfo.m_MagicStringLength + 2);

        if (!ifStream)
        {
            throw armnn::Exception(
                    fmt::format("Failed to create numpy header info at {}",
                                CHECK_LOCATION().AsString()));
        }
        // Verify that the numpy is in the valid format by checking for the magic string
        int compare_result = ::memcmp(buffer, headerInfo.m_MagicString, headerInfo.m_MagicStringLength);
        if (compare_result != 0) {
            throw armnn::Exception(fmt::format("Numpy does not contain magic string. Can not parse invalid numpy {}",
                                               CHECK_LOCATION().AsString()));
        }

        headerInfo.m_MajorVersion = buffer[headerInfo.m_MagicStringLength];
        headerInfo.m_MinorVersion = buffer[headerInfo.m_MagicStringLength + 1];
        if(headerInfo.m_MajorVersion == 1 && headerInfo.m_MinorVersion == 0)
        {
            ifStream.read(headerInfo.m_HeaderLenBytes, 2);
            // Header len is written in little endian, so we do a quick test
            // to check the machines endianness
            int i = 1;
            if (*(reinterpret_cast<char *>(&i)) == 1)
            {
                headerInfo.m_HeaderLen = static_cast<unsigned>(headerInfo.m_HeaderLenBytes[0]) |
                                         (static_cast<unsigned>(headerInfo.m_HeaderLenBytes[1] << 8));
            }
            else
            {
                headerInfo.m_HeaderLen = static_cast<unsigned>(headerInfo.m_HeaderLenBytes[1]) |
                                         (static_cast<unsigned>(headerInfo.m_HeaderLenBytes[0] << 8));
            }
        }
        else if (headerInfo.m_MajorVersion == 2 && headerInfo.m_MinorVersion == 0)
        {
            ifStream.read(headerInfo.m_HeaderLenBytes, 4);
            // Header len is written in little endian, so we do a quick test
            // to check the machines endianness
            int i = 1;
            if (*(reinterpret_cast<char *>(&i)) == 1)
            {
                headerInfo.m_HeaderLen = static_cast<unsigned>(headerInfo.m_HeaderLenBytes[0] << 0) |
                    static_cast<unsigned>(headerInfo.m_HeaderLenBytes[1] << 8) |
                    static_cast<unsigned>(headerInfo.m_HeaderLenBytes[2] << 16) |
                    static_cast<unsigned>(headerInfo.m_HeaderLenBytes[3] << 24);
            }
            else
            {
                headerInfo.m_HeaderLen = static_cast<unsigned>(headerInfo.m_HeaderLenBytes[3] << 0) |
                    static_cast<unsigned>(headerInfo.m_HeaderLenBytes[2] << 8) |
                    static_cast<unsigned>(headerInfo.m_HeaderLenBytes[1] << 16) |
                    static_cast<unsigned>(headerInfo.m_HeaderLenBytes[0] << 24);
            }
        }
        else
        {
            throw armnn::ParseException(fmt::format("Unable to parser Numpy version {}.{} {}",
                                                    headerInfo.m_MajorVersion,
                                                    headerInfo.m_MinorVersion,
                                                    CHECK_LOCATION().AsString()));
        }
    }

    /// Primarily used to isolate values from header dictionary
    inline std::string getSubstring(std::string fullString,
                                    std::string substringStart,
                                    std::string substringEnd,
                                    bool removeStartChar = 0,
                                    bool includeEndChar = 0)
    {
        size_t startPos = fullString.find(substringStart);
        size_t endPos = fullString.find(substringEnd, startPos);
        if (startPos == std::string::npos || endPos == std::string::npos)
        {
            throw armnn::ParseException(fmt::format("Unable to find {} in numpy file.",
                                                    CHECK_LOCATION().AsString()));
        }

        // std::string.substr takes the starting position and the length of the substring.
        // To calculate the length we subtract the start position from the end position.
        // We also add a boolean on whether or not we want to include the character used to find endPos
        startPos+= removeStartChar;
        endPos += includeEndChar;
        return fullString.substr(startPos, endPos - startPos);
    }

    inline void parseShape(Header& header, std::string& shapeString)
    {
        std::istringstream shapeStringStream(shapeString);
        std::string token;
        while(getline(shapeStringStream, token, ','))
        {
            header.m_Shape.push_back(static_cast<uint32_t >(std::stoi(token)));
        }
    }

    inline void CreateHeader(std::ifstream& ifStream, HeaderInfo& headerInfo, Header& header)
    {
        char stringBuffer[headerInfo.m_HeaderLen];
        ifStream.read(stringBuffer, static_cast<std::streamsize>(headerInfo.m_HeaderLen));

        header.m_HeaderString = std::string(stringBuffer, headerInfo.m_HeaderLen);
        // Remove new line character at the end of the string
        if(header.m_HeaderString.back() == '\n')
        {
            header.m_HeaderString.pop_back();
        }

        // Remove whitespace from the string.
        // std::remove shuffles the string by place all whitespace at the end and
        //     returning the start location of the shuffled whitespace.
        // std::string.erase then deletes the whitespace by deleting characters
        //     between the iterator returned from std::remove and the end of the std::string
        std::string::iterator whitespaceSubstringStart = std::remove(header.m_HeaderString.begin(),
                                                                     header.m_HeaderString.end(), ' ');
        header.m_HeaderString.erase(whitespaceSubstringStart, header.m_HeaderString.end());

        // The order of the dictionary should be alphabetical,
        // however this is not guarenteed so we have to search for the string.
        // Because of this we do some weird parsing using std::string.find and some magic patterns
        //
        // For the description value, we include the end character from the first substring
        // to help us find the value in the second substring. This should return with a "," at the end.
        // Since we previously left the "," at the end of the substring,
        // we can use it to find the end of the description value and then remove it after.
        std::string descrString = getSubstring(header.m_HeaderString, "'descr", ",", 0, 1);
        header.m_DescrString = getSubstring(descrString, ":", ",", 1);

        // Fortran order is a python boolean literal, we simply look for a comma to delimit this pair.
        // Since this is a boolean, both true and false end in an "e" without appearing in between.
        // It is not great, but it is the easiest way to find the end.
        // We have to ensure we include this e in the substring.
        // Since this is a boolean we can check if the string contains
        // either true or false and set the variable as required
        std::string fortranOrderString = getSubstring(header.m_HeaderString, "'fortran_order", ",");
        fortranOrderString = getSubstring(fortranOrderString, ":", "e", 1, 1);
        header.m_FortranOrder = fortranOrderString.find("True") != std::string::npos ? true : false;

        // The shape is a python tuple so we search for the closing bracket of the tuple.
        // We include the end character to help us isolate the value substring.
        // We can extract the inside of the tuple by searching for opening and closing brackets.
        // We can then remove the brackets isolating the inside of the tuple.
        // We then need to parse the string into a vector of unsigned integers
        std::string shapeString = getSubstring(header.m_HeaderString, "'shape", ")", 0, 1);
        shapeString = getSubstring(shapeString, "(", ")", 1, 0);
        parseShape(header, shapeString);
    }

    template<typename T>
    inline void ReadData(std::ifstream& ifStream, T* tensor, const unsigned int& numElements)
    {
        const std::streamsize bytes = static_cast<std::streamsize>(sizeof(T)) *
                                      static_cast<std::streamsize>(numElements);
        ifStream.read(reinterpret_cast<char *>(tensor), bytes);
    }


    inline armnn::DataType getArmNNDataType(std::string& descr)
    {
        if(descr.find("f4") != std::string::npos || descr.find("f8") != std::string::npos)
        {
            return armnn::DataType::Float32;
        }
        else if (descr.find("f2") != std::string::npos)
        {
            return armnn::DataType::Float16;
        }
        else if (descr.find("i8") != std::string::npos)
        {
            return armnn::DataType::Signed64;
        }
        else if (descr.find("i4") != std::string::npos)
        {
            return armnn::DataType::Signed32;
        }
        else if (descr.find("i2") != std::string::npos)
        {
            return armnn::DataType::QSymmS16;
        }
        else if (descr.find("i1") != std::string::npos)
        {
            return armnn::DataType::QSymmS8;
        }
        else if (descr.find("u1") != std::string::npos)
        {
            return armnn::DataType::QAsymmU8;
        }
        else
        {
            throw armnn::Exception(fmt::format("Numpy data type:{} not supported. {}",
                                               descr, CHECK_LOCATION().AsString()));
        }
    }

    inline std::string getNumpyDescr(armnn::DataType dType)
    {
        switch(dType)
        {
            case armnn::DataType::Float32:
                return "f" + std::to_string(sizeof(float)); // size of float can be 4 or 8
            case armnn::DataType::Float16:
                return "f2";
            case armnn::DataType::Signed64:
                return "i8";
            case armnn::DataType::Signed32:
                return "i4";
            case armnn::DataType::QSymmS16:
                return "i2";
            case armnn::DataType::QSymmS8:
            case armnn::DataType::QAsymmS8:
                return "i1";
            case armnn::DataType::QAsymmU8:
                return "u1";
            default:
                throw armnn::Exception(fmt::format("ArmNN to Numpy data type:{} not supported. {}",
                                       dType, CHECK_LOCATION().AsString()));
        }
    }

    template <typename T>
    inline bool compareCTypes(std::string& descr)
    {
        if(descr.find("f4") != std::string::npos || descr.find("f8") != std::string::npos)
        {
            return std::is_same<T, float>::value;
        }
        else if (descr.find("i8") != std::string::npos)
        {
            return std::is_same<T, int64_t>::value;
        }
        else if (descr.find("i4") != std::string::npos)
        {
            return std::is_same<T, int32_t>::value;
        }
        else if (descr.find("i2") != std::string::npos)
        {
            return std::is_same<T, int16_t>::value;
        }
        else if (descr.find("i1") != std::string::npos)
        {
            return std::is_same<T, int8_t>::value;
        }
        else if (descr.find("u1") != std::string::npos)
        {
            return std::is_same<T, uint8_t>::value;
        }
        else
        {
            throw armnn::Exception(fmt::format("Numpy data type:{} not supported. {}",
                                               descr, CHECK_LOCATION().AsString()));
        }
    }

    inline unsigned int getNumElements(Header& header)
    {
        unsigned int numEls = 1;
        for (auto dim: header.m_Shape)
        {
            numEls *= dim;
        }

        return numEls;
    }

    // Material in WriteToNumpyFile() has been reused from https://github.com/llohse/libnpy/blob/master/include/npy.hpp
    // Please see write_header() in the above file for more details.
    template<typename T>
    inline void WriteToNumpyFile(const std::string& outputTensorFileName,
                                 const T* const array,
                                 const unsigned int numElements,
                                 armnn::DataType dataType,
                                 const armnn::TensorShape& shape)
    {
        std::ofstream out(outputTensorFileName, std::ofstream::binary);

        // write header
        {
            // Setup string of tensor shape in format (x0, x1, x2, ..)
            std::string shapeStr = "(";
            for (uint32_t i = 0; i < shape.GetNumDimensions()-1; i++)
            {
                shapeStr = shapeStr + std::to_string(shape[i]) + ", ";
            }
            shapeStr = shapeStr + std::to_string(shape[shape.GetNumDimensions()-1]) + ")";

            int i = 1;
            std::string endianChar = (*(reinterpret_cast<char *>(&i))) ? "<" : ">";
            std::string dataTypeStr = getNumpyDescr(dataType);
            std::string fortranOrder = "False";
            std::string headerStr = "{'descr': '" + endianChar + dataTypeStr +
                                    "', 'fortran_order': " + fortranOrder +
                                    ", 'shape': " + shapeStr + ", }";

            armnnNumpy::HeaderInfo headerInfo;

            // Header is composed of:
            //     - 6 byte magic string
            //     - 2 byte major and minor version
            //     - 2 byte (v1.0) / 4 byte (v2.0) little-endian unsigned int
            //     - headerStr.length() bytes
            //     - 1 byte for newline termination (\n)
            size_t length = headerInfo.m_MagicStringLength + 2 + 2 + headerStr.length() + 1;
            uint8_t major_version = 1;

            // for numpy major version 2, add extra 2 bytes for little-endian int (total 4 bytes)
            if (length >= 255 * 255)
            {
                length += 2;
                major_version = 2;
            }

            // Pad with spaces so header length is modulo 16 bytes.
            size_t padding_length = 16 - length % 16;
            std::string padding(padding_length, ' ');

            // write magic string
            out.write(headerInfo.m_MagicString, headerInfo.m_MagicStringLength);
            out.put(static_cast<char>(major_version));
            out.put(0); // minor version

            // write header length
            if (major_version == 1)
            {
                auto header_len = static_cast<uint16_t>(headerStr.length() + padding.length() + 1);

                std::array<uint8_t, 2> header_len_16{static_cast<uint8_t>((header_len >> 0) & 0xff),
                                                     static_cast<uint8_t>((header_len >> 8) & 0xff)};
                out.write(reinterpret_cast<char *>(header_len_16.data()), 2);
            }
            else
            {
                auto header_len = static_cast<uint32_t>(headerStr.length() + padding.length() + 1);

                std::array<uint8_t, 4> header_len_32{
                    static_cast<uint8_t>((header_len >> 0) & 0xff), static_cast<uint8_t>((header_len >> 8) & 0xff),
                    static_cast<uint8_t>((header_len >> 16) & 0xff), static_cast<uint8_t>((header_len >> 24) & 0xff)};
                out.write(reinterpret_cast<char *>(header_len_32.data()), 4);
            }

            out << headerStr << padding << '\n';
        }

        // write tensor data to file
        out.write(reinterpret_cast<const char *>(array), sizeof(T) * numElements);
    }
}

#endif // NUMPY_HPP