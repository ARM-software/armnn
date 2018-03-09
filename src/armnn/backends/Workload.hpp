//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "WorkloadData.hpp"
#include "WorkloadInfo.hpp"
#include <algorithm>
#include "Profiling.hpp"

namespace armnn
{

// Workload interface to enqueue a layer computation
class IWorkload
{
public:
    virtual ~IWorkload(){};

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

template <typename QueueDescriptor, armnn::DataType DataType>
class TypedWorkload : public BaseWorkload<QueueDescriptor>
{
public:

    TypedWorkload(const QueueDescriptor& descriptor, const WorkloadInfo& info)
        : BaseWorkload<QueueDescriptor>(descriptor, info)
    {
        BOOST_ASSERT_MSG(std::all_of(info.m_InputTensorInfos.begin(),
                                     info.m_InputTensorInfos.end(),
                                     [&](auto it){
                                         return it.GetDataType() == DataType;
                                     }),
                         "Trying to create workload with incorrect type");
        BOOST_ASSERT_MSG(std::all_of(info.m_OutputTensorInfos.begin(),
                                     info.m_OutputTensorInfos.end(),
                                     [&](auto it){
                                         return it.GetDataType() == DataType;
                                     }),
                         "Trying to create workload with incorrect type");
    }

    static constexpr armnn::DataType ms_DataType = DataType;
};

template <typename QueueDescriptor>
using Float32Workload = TypedWorkload<QueueDescriptor, armnn::DataType::Float32>;

template <typename QueueDescriptor>
using Uint8Workload = TypedWorkload<QueueDescriptor, armnn::DataType::QuantisedAsymm8>;

} //namespace armnn
