//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "armnn/Tensor.hpp"
#include "armnn/Types.hpp"
#include "Network.hpp"
#include "LayerFwd.hpp"
#include "backends/RefWorkloadFactory.hpp"
#include "backends/NeonWorkloadFactory.hpp"
#include "backends/ClWorkloadFactory.hpp"
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

class LoadedNetwork
{
public:
    TensorInfo GetInputTensorInfo(LayerBindingId layerId) const;
    TensorInfo GetOutputTensorInfo(LayerBindingId layerId) const;

    Status EnqueueWorkload(const InputTensors& inputTensors, const OutputTensors& outputTensors);

    static std::unique_ptr<LoadedNetwork> MakeLoadedNetwork(std::unique_ptr<OptimizedNetwork> net,
                                                            bool useCpuRefAsFallback);

private:
    LoadedNetwork(std::unique_ptr<OptimizedNetwork> net, bool useCpuRefAsFallback);

    void EnqueueInput(const BindableLayer& layer, ITensorHandle* tensorHandle, const TensorInfo& tensorInfo);

    void EnqueueOutput(const BindableLayer& layer, ITensorHandle* tensorHandle, const TensorInfo& tensorInfo);

    bool Execute();

    void TidyWorkloadQueue(size_t numInputs, size_t numOutputs);

    const IWorkloadFactory& GetWorkloadFactory(const Layer& layer) const;

    RefWorkloadFactory  m_CpuRef;
    NeonWorkloadFactory m_CpuAcc;
    ClWorkloadFactory   m_GpuAcc;

    std::unique_ptr<OptimizedNetwork> m_OptimizedNetwork;
    std::vector< std::unique_ptr<IWorkload> > m_WorkloadQueue;
};

}
