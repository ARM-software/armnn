//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnn
{

namespace profiling
{

class IBufferWrapper
{
public:
    virtual ~IBufferWrapper() {}

    virtual unsigned char* Reserve(unsigned int requestedSize, unsigned int& reservedSize) = 0;

    virtual void Commit(unsigned int size) = 0;

    virtual const unsigned char* GetReadBuffer(unsigned int& size) = 0;

    virtual void Release(unsigned int size) = 0;
};

} // namespace profiling

} // namespace armnn
