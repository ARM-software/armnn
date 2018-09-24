//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "OutputHandler.hpp"
#include <backends/aclCommon/ArmComputeTensorUtils.hpp>

#include <arm_compute/runtime/CL/CLTensor.h>
#include <arm_compute/runtime/CL/CLSubTensor.h>
#include <arm_compute/runtime/CL/CLMemoryGroup.h>
#include <arm_compute/runtime/IMemoryGroup.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Coordinates.h>

#include <boost/polymorphic_pointer_cast.hpp>

namespace armnn
{


class IClTensorHandle : public ITensorHandle
{
public:
    virtual arm_compute::ICLTensor& GetTensor() = 0;
    virtual arm_compute::ICLTensor const& GetTensor() const = 0;
    virtual arm_compute::DataType GetDataType() const = 0;
    virtual void SetMemoryGroup(const std::shared_ptr<arm_compute::IMemoryGroup>& memoryGroup) = 0;
};

class ClTensorHandle : public IClTensorHandle
{
public:
    ClTensorHandle(const TensorInfo& tensorInfo)
    {
        armnn::armcomputetensorutils::BuildArmComputeTensor(m_Tensor, tensorInfo);
    }

    ClTensorHandle(const TensorInfo& tensorInfo, DataLayout dataLayout)
    {
        armnn::armcomputetensorutils::BuildArmComputeTensor(m_Tensor, tensorInfo, dataLayout);
    }

    arm_compute::CLTensor& GetTensor() override { return m_Tensor; }
    arm_compute::CLTensor const& GetTensor() const override { return m_Tensor; }
    virtual void Allocate() override {armnn::armcomputetensorutils::InitialiseArmComputeTensorEmpty(m_Tensor);}

    virtual void Manage() override
    {
        assert(m_MemoryGroup != nullptr);
        m_MemoryGroup->manage(&m_Tensor);
    }

    virtual const void* Map(bool blocking = true) const override
    {
        const_cast<arm_compute::CLTensor*>(&m_Tensor)->map(blocking);
        return static_cast<const void*>(m_Tensor.buffer() + m_Tensor.info()->offset_first_element_in_bytes());
    }
    virtual void Unmap() const override { const_cast<arm_compute::CLTensor*>(&m_Tensor)->unmap(); }

    virtual ITensorHandle::Type GetType() const override { return ITensorHandle::CL; }

    virtual ITensorHandle* GetParent() const override { return nullptr; }

    virtual arm_compute::DataType GetDataType() const override
    {
        return m_Tensor.info()->data_type();
    }

    virtual void SetMemoryGroup(const std::shared_ptr<arm_compute::IMemoryGroup>& memoryGroup) override
    {
        m_MemoryGroup = boost::polymorphic_pointer_downcast<arm_compute::CLMemoryGroup>(memoryGroup);
    }

    TensorShape GetStrides() const override
    {
        return armcomputetensorutils::GetStrides(m_Tensor.info()->strides_in_bytes());
    }

    TensorShape GetShape() const override
    {
        return armcomputetensorutils::GetShape(m_Tensor.info()->tensor_shape());
    }
private:
    arm_compute::CLTensor m_Tensor;
    std::shared_ptr<arm_compute::CLMemoryGroup> m_MemoryGroup;
};

class ClSubTensorHandle : public IClTensorHandle
{
public:
    ClSubTensorHandle(IClTensorHandle* parent,
                      const arm_compute::TensorShape& shape,
                      const arm_compute::Coordinates& coords)
    : m_Tensor(&parent->GetTensor(), shape, coords)
    {
        parentHandle = parent;
    }

    arm_compute::CLSubTensor& GetTensor() override { return m_Tensor; }
    arm_compute::CLSubTensor const& GetTensor() const override { return m_Tensor; }

    virtual void Allocate() override {}
    virtual void Manage() override {}

    virtual const void* Map(bool blocking = true) const override
    {
        const_cast<arm_compute::CLSubTensor*>(&m_Tensor)->map(blocking);
        return static_cast<const void*>(m_Tensor.buffer() + m_Tensor.info()->offset_first_element_in_bytes());
    }
    virtual void Unmap() const override { const_cast<arm_compute::CLSubTensor*>(&m_Tensor)->unmap(); }

    virtual ITensorHandle::Type GetType() const override { return ITensorHandle::CL; }

    virtual ITensorHandle* GetParent() const override { return parentHandle; }

    virtual arm_compute::DataType GetDataType() const override
    {
        return m_Tensor.info()->data_type();
    }

    virtual void SetMemoryGroup(const std::shared_ptr<arm_compute::IMemoryGroup>&) override {}

    TensorShape GetStrides() const override
    {
        return armcomputetensorutils::GetStrides(m_Tensor.info()->strides_in_bytes());
    }

    TensorShape GetShape() const override
    {
        return armcomputetensorutils::GetShape(m_Tensor.info()->tensor_shape());
    }

private:
    mutable arm_compute::CLSubTensor m_Tensor;
    ITensorHandle* parentHandle = nullptr;

};

}
