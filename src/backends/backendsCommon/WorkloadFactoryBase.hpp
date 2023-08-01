//
// Copyright © 2019, 2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

class WorkloadFactoryBase : public IWorkloadFactory
{
public:
    bool SupportsSubTensors() const override
    { return false; };

    std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& /*parent*/,
                                                         TensorShape const& /*subTensorShape*/,
                                                         unsigned int const */*subTensorOrigin*/) const override
    { return nullptr; };


    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& /*tensorInfo*/,
                                                      const bool /*IsMemoryManaged*/) const override
    { return nullptr; }

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& /*tensorInfo*/,
                                                      DataLayout /*dataLayout*/,
                                                      const bool /*IsMemoryManaged*/) const override
    { return nullptr; }

    std::unique_ptr<IWorkload> CreateWorkload(LayerType /*type*/,
                                              const QueueDescriptor& /*descriptor*/,
                                              const WorkloadInfo& /*info*/) const override
    { return nullptr; }

};

} //namespace armnn