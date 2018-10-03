//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "WorkloadData.hpp"
#include "WorkloadInfo.hpp"
#include <algorithm>
#include "Profiling.hpp"

namespace armnn
{

/// Workload interface to enqueue a layer computation.
class IWorkload
{
public:
    virtual ~IWorkload() {}

    virtual void Execute() const = 0;
};

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
        : m_Data(descriptor)
    {
        m_Data.Validate(info);
    }

    const QueueDescriptor& GetData() const { return m_Data; }

protected:
    const QueueDescriptor m_Data;
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
                BOOST_ASSERT_MSG(false, "Trying to create workload with incorrect type");
            }
            BOOST_ASSERT_MSG(std::all_of(std::next(info.m_InputTensorInfos.begin()),
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
                    BOOST_ASSERT_MSG(false, "Trying to create workload with incorrect type");
                }
            }
            else if (std::find(dataTypes.begin(), dataTypes.end(), expectedOutputType) == dataTypes.end())
            {
                BOOST_ASSERT_MSG(false, "Trying to create workload with incorrect type");
            }
            BOOST_ASSERT_MSG(std::all_of(std::next(info.m_OutputTensorInfos.begin()),
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
        BOOST_ASSERT_MSG(std::all_of(info.m_InputTensorInfos.begin(),
                                     info.m_InputTensorInfos.end(),
                                     [&](auto it){
                                         return it.GetDataType() == InputDataType;
                                     }),
                         "Trying to create workload with incorrect type");
        BOOST_ASSERT_MSG(std::all_of(info.m_OutputTensorInfos.begin(),
                                     info.m_OutputTensorInfos.end(),
                                     [&](auto it){
                                         return it.GetDataType() == OutputDataType;
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
using Uint8Workload = TypedWorkload<QueueDescriptor, armnn::DataType::QuantisedAsymm8>;

template <typename QueueDescriptor>
using Float16ToFloat32Workload = MultiTypedWorkload<QueueDescriptor,
                                                    armnn::DataType::Float16,
                                                    armnn::DataType::Float32>;

template <typename QueueDescriptor>
using Float32ToFloat16Workload = MultiTypedWorkload<QueueDescriptor,
                                                    armnn::DataType::Float32,
                                                    armnn::DataType::Float16>;

} //namespace armnn
