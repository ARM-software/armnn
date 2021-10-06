//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <common/include/ProfilingGuid.hpp>

namespace armnn
{

class OptimizedNetworkImpl
{
public:
    OptimizedNetworkImpl(const OptimizedNetworkImpl& other, const ModelOptions& modelOptions);
    OptimizedNetworkImpl(std::unique_ptr<Graph> graph);
    OptimizedNetworkImpl(std::unique_ptr<Graph> graph, const ModelOptions& modelOptions);
    virtual ~OptimizedNetworkImpl();

    virtual Status PrintGraph();
    virtual Status SerializeToDot(std::ostream& stream) const;

    virtual profiling::ProfilingGuid GetGuid() const { return m_Guid; };

    virtual size_t GetNumInputs() const;
    virtual size_t GetNumOutputs() const;

    Graph& GetGraph() { return *m_Graph; }
    ModelOptions& GetModelOptions() { return m_ModelOptions; }

private:
    std::unique_ptr<Graph> m_Graph;
    profiling::ProfilingGuid m_Guid;
    ModelOptions m_ModelOptions;
};

}
