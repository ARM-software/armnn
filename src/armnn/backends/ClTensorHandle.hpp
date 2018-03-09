//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "OutputHandler.hpp"
#include "ArmComputeTensorUtils.hpp"

#include <arm_compute/runtime/CL/CLTensor.h>
#include <arm_compute/runtime/CL/CLSubTensor.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Coordinates.h>


namespace armnn
{


class IClTensorHandle : public ITensorHandle
{
public:
    virtual arm_compute::ICLTensor& GetTensor() = 0;
    virtual arm_compute::ICLTensor const& GetTensor() const = 0;
    virtual void Map(bool blocking = true) = 0;
    virtual void UnMap() = 0;
    virtual arm_compute::DataType GetDataType() const = 0;
};

class ClTensorHandle : public IClTensorHandle
{
public:
    ClTensorHandle(const TensorInfo& tensorInfo)
    {
        armnn::armcomputetensorutils::BuildArmComputeTensor(m_Tensor, tensorInfo);
    }

    arm_compute::CLTensor& GetTensor() override { return m_Tensor; }
    arm_compute::CLTensor const& GetTensor() const override { return m_Tensor; }
    virtual void Allocate() override {armnn::armcomputetensorutils::InitialiseArmComputeTensorEmpty(m_Tensor);};

    virtual void Map(bool blocking = true) override {m_Tensor.map(blocking);}
    virtual void UnMap() override { m_Tensor.unmap();}

    virtual ITensorHandle::Type GetType() const override { return ITensorHandle::CL;}

    virtual arm_compute::DataType GetDataType() const override
    {
        return m_Tensor.info()->data_type();
    }

private:
    arm_compute::CLTensor m_Tensor;

};

class ClSubTensorHandle : public IClTensorHandle
{
public:
    ClSubTensorHandle(arm_compute::ICLTensor& parent,
                   const arm_compute::TensorShape& shape,
                   const arm_compute::Coordinates& coords)
    : m_Tensor(&parent, shape, coords)
    {
    }

    arm_compute::CLSubTensor& GetTensor() override { return m_Tensor; }
    arm_compute::CLSubTensor const& GetTensor() const override { return m_Tensor; }
    virtual void Allocate() override {};

    virtual void Map(bool blocking = true) override {m_Tensor.map(blocking);}
    virtual void UnMap() override { m_Tensor.unmap();}

    virtual ITensorHandle::Type GetType() const override { return ITensorHandle::CL;}

    virtual arm_compute::DataType GetDataType() const override
    {
        return m_Tensor.info()->data_type();
    }

private:
    arm_compute::CLSubTensor m_Tensor;

};

}