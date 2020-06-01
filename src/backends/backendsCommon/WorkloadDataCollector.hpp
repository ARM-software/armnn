//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Tensor.hpp>

#include <vector>

namespace armnn
{
class ITensorHandle;

class WorkloadDataCollector
{
public:
    WorkloadDataCollector(std::vector<ITensorHandle*>& handles, std::vector<TensorInfo>& infos)
        : m_Handles(handles)
        , m_Infos(infos)
    {
    }

    void Push(ITensorHandle* handle, const TensorInfo& info)
    {
        m_Handles.push_back(handle);
        m_Infos.push_back(info);
    }

private:
    std::vector<ITensorHandle*>& m_Handles;
    std::vector<TensorInfo>& m_Infos;
};


} //namespace armnn
