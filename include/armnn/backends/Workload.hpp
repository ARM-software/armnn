//
// Copyright © 2021-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "IWorkload.hpp"
#include "WorkloadData.hpp"
#include "WorkloadInfo.hpp"

#include <armnn/Logging.hpp>

#include <Profiling.hpp>

#include <client/include/IProfilingService.hpp>

#include <algorithm>

namespace armnn
{

// NullWorkload used to denote an unsupported workload when used by the MakeWorkload<> template
// in the various workload factories.
// There should never be an instantiation of a NullWorkload.
class NullWorkload : public IWorkload
{
    NullWorkload()=delete;
};

template <typename QueueDescriptor>
class BaseWorkload : public IWorkload
{
public:

    BaseWorkload(const QueueDescriptor& descriptor, const WorkloadInfo& info)
        : m_Data(descriptor),
          m_Guid(arm::pipe::IProfilingService::GetNextGuid()),
          m_Name(info.m_Name)
    {
        m_Data.Validate(info);
    }

    virtual const std::string& GetName() const override
    {
        return m_Name;
    }

    void PostAllocationConfigure() override {}

    const QueueDescriptor& GetData() const { return m_Data; }

    arm::pipe::ProfilingGuid GetGuid() const final { return m_Guid; }

    virtual bool SupportsTensorHandleReplacement()  const override
    {
        return false;
    }

    // Replace input tensor handle with the given TensorHandle
    void ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override
    {
        armnn::IgnoreUnused(tensorHandle, slot);
        throw armnn::UnimplementedException("ReplaceInputTensorHandle not implemented for this workload");
    }

    // Replace output tensor handle with the given TensorHandle
    void ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override
    {
        armnn::IgnoreUnused(tensorHandle, slot);
        throw armnn::UnimplementedException("ReplaceOutputTensorHandle not implemented for this workload");
    }

protected:
    QueueDescriptor m_Data;
    const arm::pipe::ProfilingGuid m_Guid;
    const std::string m_Name;
};

// TypedWorkload used
template <typename QueueDescriptor, armnn::DataType... DataTypes>
class TypedWorkload : public BaseWorkload<QueueDescriptor>
{
public:

    TypedWorkload(const QueueDescriptor& descriptor, const WorkloadInfo& info)
        : BaseWorkload<QueueDescriptor>(descriptor, info)
    {
        std::vector<armnn::DataType> dataTypes = {DataTypes...};
        armnn::DataType expectedInputType;

        if (!info.m_InputTensorInfos.empty())
        {
            expectedInputType = info.m_InputTensorInfos.front().GetDataType();

            if (std::find(dataTypes.begin(), dataTypes.end(), expectedInputType) == dataTypes.end())
            {
                throw armnn::Exception("Trying to create workload with incorrect type");
            }
            if (std::all_of(std::next(info.m_InputTensorInfos.begin()),
                                         info.m_InputTensorInfos.end(),
                                         [&](auto it){
                                             return it.GetDataType() == expectedInputType;
                                         }) == false)
            {
                throw armnn::Exception("Trying to create workload with incorrect type");
            }
        }
        armnn::DataType expectedOutputType;

        if (!info.m_OutputTensorInfos.empty())
        {
            expectedOutputType = info.m_OutputTensorInfos.front().GetDataType();

            if (!info.m_InputTensorInfos.empty())
            {
                expectedInputType = info.m_InputTensorInfos.front().GetDataType();

                if (expectedOutputType != expectedInputType)
                {
                    throw armnn::Exception( "Trying to create workload with incorrect type");
                }
            }
            else if (std::find(dataTypes.begin(), dataTypes.end(), expectedOutputType) == dataTypes.end())
            {
                throw armnn::Exception("Trying to create workload with incorrect type");
            }
            if (std::all_of(std::next(info.m_OutputTensorInfos.begin()),
                                         info.m_OutputTensorInfos.end(),
                                         [&](auto it){
                                             return it.GetDataType() == expectedOutputType;
                                         }) == false)
            {
                throw armnn::Exception("Trying to create workload with incorrect type");
            }
        }
    }
};

template <typename QueueDescriptor, armnn::DataType InputDataType, armnn::DataType OutputDataType>
class MultiTypedWorkload : public BaseWorkload<QueueDescriptor>
{
public:

    MultiTypedWorkload(const QueueDescriptor& descriptor, const WorkloadInfo& info)
        : BaseWorkload<QueueDescriptor>(descriptor, info)
    {
        if (std::all_of(info.m_InputTensorInfos.begin(),
                                     info.m_InputTensorInfos.end(),
                                     [&](auto it){
                                         return it.GetDataType() == InputDataType;
                                     }) == false)
        {
            throw armnn::Exception("Trying to create workload with incorrect type");
        }
        if (std::all_of(info.m_OutputTensorInfos.begin(),
                                     info.m_OutputTensorInfos.end(),
                                     [&](auto it){
                                         return it.GetDataType() == OutputDataType;
                                     }) == false)
        {
            throw armnn::Exception("Trying to create workload with incorrect type");
        }
    }
};

// FirstInputTypedWorkload used to check type of the first input
template <typename QueueDescriptor, armnn::DataType DataType>
class FirstInputTypedWorkload : public BaseWorkload<QueueDescriptor>
{
public:

    FirstInputTypedWorkload(const QueueDescriptor& descriptor, const WorkloadInfo& info)
        : BaseWorkload<QueueDescriptor>(descriptor, info)
    {
        if (!info.m_InputTensorInfos.empty())
        {
            if (info.m_InputTensorInfos.front().GetDataType() != DataType)
            {
                throw armnn::Exception("Trying to create workload with incorrect type");
            }
        }

        if (std::all_of(info.m_OutputTensorInfos.begin(),
                                     info.m_OutputTensorInfos.end(),
                                     [&](auto it){
                                         return it.GetDataType() == DataType;
                                     }) == false)
        {
            throw armnn::Exception("Trying to create workload with incorrect type");
        }
    }
};

template <typename QueueDescriptor>
using FloatWorkload = TypedWorkload<QueueDescriptor,
                                    armnn::DataType::Float16,
                                    armnn::DataType::Float32>;

template <typename QueueDescriptor>
using Float32Workload = TypedWorkload<QueueDescriptor, armnn::DataType::Float32>;

template <typename QueueDescriptor>
using Uint8Workload = TypedWorkload<QueueDescriptor, armnn::DataType::QAsymmU8>;

template <typename QueueDescriptor>
using Int32Workload = TypedWorkload<QueueDescriptor, armnn::DataType::Signed32>;

template <typename QueueDescriptor>
using BooleanWorkload = TypedWorkload<QueueDescriptor, armnn::DataType::Boolean>;

template <typename QueueDescriptor>
using BaseFloat32ComparisonWorkload = MultiTypedWorkload<QueueDescriptor,
                                                         armnn::DataType::Float32,
                                                         armnn::DataType::Boolean>;

template <typename QueueDescriptor>
using BaseUint8ComparisonWorkload = MultiTypedWorkload<QueueDescriptor,
                                                       armnn::DataType::QAsymmU8,
                                                       armnn::DataType::Boolean>;

template <typename QueueDescriptor>
using BFloat16ToFloat32Workload = MultiTypedWorkload<QueueDescriptor,
                                                     armnn::DataType::BFloat16,
                                                     armnn::DataType::Float32>;

template <typename QueueDescriptor>
using Float32ToBFloat16Workload = MultiTypedWorkload<QueueDescriptor,
                                                     armnn::DataType::Float32,
                                                     armnn::DataType::BFloat16>;

template <typename QueueDescriptor>
using Float16ToFloat32Workload = MultiTypedWorkload<QueueDescriptor,
                                                    armnn::DataType::Float16,
                                                    armnn::DataType::Float32>;

template <typename QueueDescriptor>
using Float32ToFloat16Workload = MultiTypedWorkload<QueueDescriptor,
                                                    armnn::DataType::Float32,
                                                    armnn::DataType::Float16>;

template <typename QueueDescriptor>
using Uint8ToFloat32Workload = MultiTypedWorkload<QueueDescriptor,
                                                  armnn::DataType::QAsymmU8,
                                                  armnn::DataType::Float32>;

} //namespace armnn
