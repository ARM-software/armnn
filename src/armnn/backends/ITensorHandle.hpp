//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

namespace armnn
{

class ITensorHandle
{
public:
    enum Type
    {
        Cpu,
        CL,
        Neon
    };

    virtual ~ITensorHandle(){}
    virtual void Allocate() = 0;
    virtual ITensorHandle::Type GetType() const = 0;
};

}
