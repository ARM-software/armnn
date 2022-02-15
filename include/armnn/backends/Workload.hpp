//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "IWorkload.hpp"
#include "WorkloadData.hpp"
#include "WorkloadInfo.hpp"
#include "WorkingMemDescriptor.hpp"

#include <Profiling.hpp>
#include <ProfilingService.hpp>

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
          m_Guid(profiling::ProfilingService::GetNextGuid())
    {
        m_Data.Validate(info);
    }

    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor) override
    {
        ARMNN_LOG(info) << "Using default async workload execution, this will network affect performance";
        std::lock_guard<std::mutex> lockGuard(m_AsyncWorkloadMutex);

        m_Data.m_Inputs = workingMemDescriptor.m_Inputs;
        m_Data.m_Outputs = workingMemDescriptor.m_Outputs;

        Execute();
    };

    void PostAllocationConfigure() override {}

    const QueueDescriptor& GetData() const { return m_Data; }

    profiling::ProfilingGuid GetGuid() const final { return m_Guid; }

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
    const profiling::ProfilingGuid m_Guid;

private:
    std::mutex m_AsyncWorkloadMutex;
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
                ARMNN_ASSERT_MSG(false, "Trying to create workload with incorrect type");
            }
            ARMNN_ASSERT_MSG(std::all_of(std::next(info.m_InputTensorInfos.begin()),
                                         info.m_InputTensorInfos.end(),
                                         [&](auto it){
                                             return it.GetDataType() == expectedInputType;
                                         }),
                             "Trying to create workload with incorrect type");
        }
        armnn::DataType expectedOutputType;

        if (!info.m_OutputTensorInfos.empty())
        {
            expectedOutputType = info.m_OutputTensorInfos.front().GetDataType();

            if (!info.m_InputTensorInfos.empty())
            {
                if (expectedOutputType != expectedInputType)
                {
                    ARMNN_ASSERT_MSG(false, "Trying to create workload with incorrect type");
                }
            }
            else if (std::find(dataTypes.begin(), dataTypes.end(), expectedOutputType) == dataTypes.end())
            {
                ARMNN_ASSERT_MSG(false, "Trying to create workload with incorrect type");
            }
            ARMNN_ASSERT_MSG(std::all_of(std::next(info.m_OutputTensorInfos.begin()),
                                         info.m_OutputTensorInfos.end(),
                                         [&](auto it){
                                             return it.GetDataType() == expectedOutputType;
                                         }),
                             "Trying to create workload with incorrect type");
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
        ARMNN_ASSERT_MSG(std::all_of(info.m_InputTensorInfos.begin(),
                                     info.m_InputTensorInfos.end(),
                                     [&](auto it){
                                         return it.GetDataType() == InputDataType;
                                     }),
                         "Trying to create workload with incorrect type");

        ARMNN_ASSERT_MSG(std::all_of(info.m_OutputTensorInfos.begin(),
                                     info.m_OutputTensorInfos.end(),
                                     [&](auto it){
                                         return it.GetDataType() == OutputDataType;
                                     }),
                         "Trying to create workload with incorrect type");
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
            ARMNN_ASSERT_MSG(info.m_InputTensorInfos.front().GetDataType() == DataType,
                                 "Trying to create workload with incorrect type");
        }

        ARMNN_ASSERT_MSG(std::all_of(info.m_OutputTensorInfos.begin(),
                                     info.m_OutputTensorInfos.end(),
                                     [&](auto it){
                                         return it.GetDataType() == DataType;
                                     }),
                         "Trying to create workload with incorrect type");
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
