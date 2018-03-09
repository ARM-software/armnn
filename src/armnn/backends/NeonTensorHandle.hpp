//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "OutputHandler.hpp"
#include "ArmComputeTensorUtils.hpp"

#include <arm_compute/runtime/Tensor.h>
#include <arm_compute/runtime/SubTensor.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Coordinates.h>


namespace armnn
{

class INeonTensorHandle : public ITensorHandle
{
public:
    virtual arm_compute::ITensor& GetTensor() = 0;
    virtual arm_compute::ITensor const& GetTensor() const = 0;
    virtual arm_compute::DataType GetDataType() const = 0;
};

class NeonTensorHandle : public INeonTensorHandle
{
public:
    NeonTensorHandle(const TensorInfo& tensorInfo)
    {
        armnn::armcomputetensorutils::BuildArmComputeTensor(m_Tensor, tensorInfo);
    }

    arm_compute::ITensor& GetTensor() override { return m_Tensor; }
    arm_compute::ITensor const& GetTensor() const override { return m_Tensor; }
    virtual void Allocate() override
    {
        armnn::armcomputetensorutils::InitialiseArmComputeTensorEmpty(m_Tensor);
    };

    virtual ITensorHandle::Type GetType() const override { return ITensorHandle::Neon; }

    virtual arm_compute::DataType GetDataType() const override
    {
        return m_Tensor.info()->data_type();
    }

private:
    arm_compute::Tensor m_Tensor;
};

class NeonSubTensorHandle : public INeonTensorHandle
{
public:
    NeonSubTensorHandle(arm_compute::ITensor& parent,
        const arm_compute::TensorShape& shape,
        const arm_compute::Coordinates& coords)
     : m_Tensor(&parent, shape, coords)
    {
    }

    arm_compute::ITensor& GetTensor() override { return m_Tensor; }
    arm_compute::ITensor const& GetTensor() const override { return m_Tensor; }
    virtual void Allocate() override
    {
    };

    virtual ITensorHandle::Type GetType() const override { return ITensorHandle::Neon; }

    virtual arm_compute::DataType GetDataType() const override
    {
        return m_Tensor.info()->data_type();
    }

private:
    arm_compute::SubTensor m_Tensor;   
};

}
