//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Filesystem.hpp"

#if defined(__unix__)
#include <sys/stat.h>
#include <stdio.h>
#elif defined(_MSC_VER)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

namespace armnnUtils
{
namespace Filesystem
{

long long GetFileSize(const char* path)
{
#if defined(__ANDROID__)
    struct stat statusBuffer;
    if (stat(path, & statusBuffer) != 0)
    {
        return -1;
    }
    return statusBuffer.st_size;
#elif defined(__unix__)
    struct stat statusBuffer;
    if (stat(path, & statusBuffer) != 0)
    {
        return -1;
    }
    return static_cast<long long>(statusBuffer.st_size);
#elif defined(_MSC_VER)
    WIN32_FILE_ATTRIBUTE_DATA attr;
    if (::GetFileAttributesEx(path, GetFileExInfoStandard, &attr) == 0)
    {
        return -1;
    }
    return attr.nFileSizeLow;
#endif
}

bool Remove(const char* path)
{
#if defined(__unix__)
    return remove(path) == 0;
#elif defined(_MSC_VER)
    return ::DeleteFile(path);
#endif
}

}
}
