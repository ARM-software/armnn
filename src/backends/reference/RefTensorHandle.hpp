//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <backendsCommon/CpuTensorHandle.hpp>

namespace armnn
{

// An implementation of ITensorHandle with simple "bump the pointer" memory-management behaviour
class RefTensorHandle : public ITensorHandle
{
public:
    RefTensorHandle(const TensorInfo& tensorInfo);

    ~RefTensorHandle();

    virtual void Manage() override
    {}

    virtual ITensorHandle* GetParent() const override
    {
        return nullptr;
    }

    virtual const void* Map(bool /* blocking = true */) const override
    {
        return m_Memory;
    }

    virtual void Unmap() const override
    {}

    virtual void Allocate() override;

    TensorShape GetStrides() const override
    {
        return GetUnpaddedTensorStrides(m_TensorInfo);
    }

    TensorShape GetShape() const override
    {
        return m_TensorInfo.GetShape();
    }

    const TensorInfo& GetTensorInfo() const
    {
        return m_TensorInfo;
    }

private:
    // Only used for testing
    void CopyOutTo(void*) const override;
    void CopyInFrom(const void*) override;

    RefTensorHandle(const RefTensorHandle& other) = delete;

    RefTensorHandle& operator=(const RefTensorHandle& other) = delete;

    TensorInfo m_TensorInfo;
    void* m_Memory;
};

}