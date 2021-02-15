//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

namespace armnn
{

class OptimizedNetworkImpl
{
public:
    OptimizedNetworkImpl(std::unique_ptr<Graph> graph);
    OptimizedNetworkImpl(std::unique_ptr<Graph> graph, const ModelOptions& modelOptions);
    virtual ~OptimizedNetworkImpl();

    virtual Status PrintGraph();
    virtual Status SerializeToDot(std::ostream& stream) const;

    virtual profiling::ProfilingGuid GetGuid() const { return m_Guid; };

    Graph& GetGraph() { return *m_Graph; }
    ModelOptions& GetModelOptions() { return m_ModelOptions; }

private:
    std::unique_ptr<Graph> m_Graph;
    profiling::ProfilingGuid m_Guid;
    ModelOptions m_ModelOptions;
};

}
