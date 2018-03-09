//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "armnn/Tensor.hpp"
#include "armnn/Types.hpp"
#include "Network.hpp"
#include "LayerFwd.hpp"
#include "backends/Workload.hpp"
#include "backends/WorkloadFactory.hpp"

namespace cl
{
    class Context;
    class CommandQueue;
    class Device;
}

namespace armnn
{

struct WorkloadFactories;

class LoadedNetwork
{
public:
    TensorInfo GetInputTensorInfo(LayerBindingId layerId) const;
    TensorInfo GetOutputTensorInfo(LayerBindingId layerId) const;

    Status EnqueueWorkload(const InputTensors& inputTensors, const OutputTensors& outputTensors,
        const WorkloadFactories& workloadFactories);

    static std::unique_ptr<LoadedNetwork> MakeLoadedNetwork(std::unique_ptr<OptimizedNetwork> net,
        const WorkloadFactories& workloadFactories);

private:
    LoadedNetwork(std::unique_ptr<OptimizedNetwork> net, const WorkloadFactories& workloadFactories);

    void EnqueueInput(const BindableLayer& layer, ITensorHandle* tensorHandle, const TensorInfo& tensorInfo,
        const WorkloadFactories& workloadFactories);

    void EnqueueOutput(const BindableLayer& layer, ITensorHandle* tensorHandle,
        const TensorInfo& tensorInfo, const WorkloadFactories& workloadFactories);

    bool Execute();

    void TidyWorkloadQueue(size_t numInputs, size_t numOutputs);

    const std::shared_ptr<IWorkloadFactory> GetWorkloadFactory(const Layer& layer,
        const WorkloadFactories& workloadFactories) const;

    std::unique_ptr<OptimizedNetwork> m_OptimizedNetwork;

    std::vector< std::unique_ptr<IWorkload> > m_WorkloadQueue;
};

}
