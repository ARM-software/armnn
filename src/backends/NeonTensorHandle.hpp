//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "OutputHandler.hpp"
#include <backends/aclCommon/ArmComputeTensorUtils.hpp>

#include <arm_compute/runtime/MemoryGroup.h>
#include <arm_compute/runtime/IMemoryGroup.h>
#include <arm_compute/runtime/Tensor.h>
#include <arm_compute/runtime/SubTensor.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Coordinates.h>

#include <boost/polymorphic_pointer_cast.hpp>

namespace armnn
{

class INeonTensorHandle : public ITensorHandle
{
public:
    virtual arm_compute::ITensor& GetTensor() = 0;
    virtual arm_compute::ITensor const& GetTensor() const = 0;
    virtual arm_compute::DataType GetDataType() const = 0;
    virtual void SetMemoryGroup(const std::shared_ptr<arm_compute::IMemoryGroup>& memoryGroup) = 0;
};

class NeonTensorHandle : public INeonTensorHandle
{
public:
    NeonTensorHandle(const TensorInfo& tensorInfo)
    {
        armnn::armcomputetensorutils::BuildArmComputeTensor(m_Tensor, tensorInfo);
    }

    NeonTensorHandle(const TensorInfo& tensorInfo, DataLayout dataLayout)
    {
        armnn::armcomputetensorutils::BuildArmComputeTensor(m_Tensor, tensorInfo, dataLayout);
    }

    arm_compute::ITensor& GetTensor() override { return m_Tensor; }
    arm_compute::ITensor const& GetTensor() const override { return m_Tensor; }

    virtual void Allocate() override
    {
        armnn::armcomputetensorutils::InitialiseArmComputeTensorEmpty(m_Tensor);
    };

    virtual void Manage() override
    {
        BOOST_ASSERT(m_MemoryGroup != nullptr);
        m_MemoryGroup->manage(&m_Tensor);
    }

    virtual ITensorHandle::Type GetType() const override { return ITensorHandle::Neon; }

    virtual ITensorHandle* GetParent() const override { return nullptr; }

    virtual arm_compute::DataType GetDataType() const override
    {
        return m_Tensor.info()->data_type();
    }

    virtual void SetMemoryGroup(const std::shared_ptr<arm_compute::IMemoryGroup>& memoryGroup) override
    {
        m_MemoryGroup = boost::polymorphic_pointer_downcast<arm_compute::MemoryGroup>(memoryGroup);
    }

    virtual const void* Map(bool /* blocking = true */) const override
    {
        return static_cast<const void*>(m_Tensor.buffer() + m_Tensor.info()->offset_first_element_in_bytes());
    }
    virtual void Unmap() const override {}


    TensorShape GetStrides() const override
    {
        return armcomputetensorutils::GetStrides(m_Tensor.info()->strides_in_bytes());
    }

    TensorShape GetShape() const override
    {
        return armcomputetensorutils::GetShape(m_Tensor.info()->tensor_shape());
    }

private:
    arm_compute::Tensor m_Tensor;
    std::shared_ptr<arm_compute::MemoryGroup> m_MemoryGroup;
};

class NeonSubTensorHandle : public INeonTensorHandle
{
public:
    NeonSubTensorHandle(INeonTensorHandle* parent,
                        const arm_compute::TensorShape& shape,
                        const arm_compute::Coordinates& coords)
     : m_Tensor(&parent->GetTensor(), shape, coords)
    {
        parentHandle = parent;
    }

    arm_compute::ITensor& GetTensor() override { return m_Tensor; }
    arm_compute::ITensor const& GetTensor() const override { return m_Tensor; }

    virtual void Allocate() override {}
    virtual void Manage() override {}

    virtual ITensorHandle::Type GetType() const override { return ITensorHandle::Neon; }

    virtual ITensorHandle* GetParent() const override { return parentHandle; }

    virtual arm_compute::DataType GetDataType() const override
    {
        return m_Tensor.info()->data_type();
    }

    virtual void SetMemoryGroup(const std::shared_ptr<arm_compute::IMemoryGroup>&) override {}

    virtual const void* Map(bool /* blocking = true */) const override
    {
        return static_cast<const void*>(m_Tensor.buffer() + m_Tensor.info()->offset_first_element_in_bytes());
    }
    virtual void Unmap() const override {}

    TensorShape GetStrides() const override
    {
        return armcomputetensorutils::GetStrides(m_Tensor.info()->strides_in_bytes());
    }

    TensorShape GetShape() const override
    {
        return armcomputetensorutils::GetShape(m_Tensor.info()->tensor_shape());
    }
private:
    arm_compute::SubTensor m_Tensor;
    ITensorHandle* parentHandle = nullptr;
};

}
