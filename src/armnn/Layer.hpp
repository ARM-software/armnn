//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "LayerFwd.hpp"

#include "backends/OutputHandler.hpp"
#include "backends/WorkloadDataCollector.hpp"
#include "backends/WorkloadInfo.hpp"
#include "InternalTypes.hpp"
#include "SerializeLayerParameters.hpp"

#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/INetwork.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/cast.hpp>

namespace armnn
{

class IWorkload;
class IWorkloadFactory;
class Layer;
class Graph;

class InputSlot final : public IInputSlot
{
public:
    explicit InputSlot(Layer& owner, unsigned int slotIndex)
    : m_OwningLayer(owner)
    , m_Connection(nullptr)
    , m_SlotIndex(slotIndex)
    {}

    ~InputSlot();

    Layer& GetOwningLayer() const { return m_OwningLayer; }
    unsigned int GetSlotIndex() const { return m_SlotIndex; }

    const OutputSlot* GetConnectedOutputSlot() const { return m_Connection; }
    OutputSlot* GetConnectedOutputSlot() { return m_Connection; }

    /// Links the slot to an output slot or breaks an existing link if passing nullptr
    void SetConnection(OutputSlot* source)
    {
        if (m_Connection != nullptr && source != nullptr)
        {
            throw InvalidArgumentException("Tried to connect an output slot to an input slot, "
                "but the latter already has a connection");
        }
        m_Connection = source;
    }

    // Insert single-output existing layer at this point in the graph.
    void Insert(Layer& layer);

    // IInputSlot

    const IOutputSlot* GetConnection() const override;
    IOutputSlot* GetConnection() override;

private:
    Layer& m_OwningLayer;
    OutputSlot* m_Connection;
    const unsigned int m_SlotIndex;
};

class OutputSlot final : public IOutputSlot
{
public:
    explicit OutputSlot(Layer& owner, OutputHandler& outputHandler)
    : m_OwningLayer(owner)
    , m_OutputHandler(outputHandler)
    {}

    ~OutputSlot()
    {
        DisconnectAll();
    }

    Layer& GetOwningLayer() const { return m_OwningLayer; }

    const OutputHandler& GetOutputHandler() const { return m_OutputHandler; }
    OutputHandler& GetOutputHandler() { return m_OutputHandler; }

    int Connect(InputSlot& destination);
    void Disconnect(InputSlot& slot);

    const std::vector<InputSlot*>& GetConnections() const { return m_Connections; }

    bool ValidateTensorShape(const TensorShape& shape) const;

    // Disconnect all conections
    void DisconnectAll();

    /// Move all connections to another OutputSlot
    void MoveAllConnections(OutputSlot& destination);

    // IOutputSlot

    unsigned int GetNumConnections() const override { return boost::numeric_cast<unsigned int>(m_Connections.size()); }
    const InputSlot* GetConnection(unsigned int index) const override;
    InputSlot* GetConnection(unsigned int index) override;

    void SetTensorInfo(const TensorInfo& tensorInfo) override;
    const TensorInfo& GetTensorInfo() const override;
    bool IsTensorInfoSet() const override;

    int Connect(IInputSlot& destination) override
    {
        return Connect(*boost::polymorphic_downcast<InputSlot*>(&destination));
    }

    void Disconnect(IInputSlot& slot) override
    {
        return Disconnect(*boost::polymorphic_downcast<InputSlot*>(&slot));
    }

private:
    void ValidateConnectionIndex(unsigned int index) const;

    Layer& m_OwningLayer;
    OutputHandler& m_OutputHandler;
    std::vector<InputSlot*> m_Connections;
};

// InputSlot inlines that need OutputSlot declaration

inline InputSlot::~InputSlot()
{
    if (m_Connection != nullptr)
    {
        m_Connection->Disconnect(*this);
    }
}

inline const IOutputSlot* InputSlot::GetConnection() const { return GetConnectedOutputSlot(); }
inline IOutputSlot* InputSlot::GetConnection() { return GetConnectedOutputSlot(); }

// Base layer class

using LayerPriority = unsigned int;

class Layer : public IConnectableLayer
{
public:
    /// @param name Optional name for the layer (may be nullptr)
    Layer(unsigned int numInputSlots, unsigned int numOutputSlots, LayerType type, const char* name);

    const std::string& GetNameStr() const
    {
        return m_LayerName;
    }

    const OutputHandler& GetOutputHandler(unsigned int i = 0) const
    {
        return m_OutputHandlers[i];
    }

    OutputHandler& GetOutputHandler(unsigned int i = 0)
    {
        return const_cast<OutputHandler&>(const_cast<const Layer*>(this)->GetOutputHandler(i));
    }

    const std::vector<InputSlot>& GetInputSlots() const { return m_InputSlots; }
    const std::vector<OutputSlot>& GetOutputSlots() const { return m_OutputSlots; }

    // Allow non-const access to input slots, but don't expose vector (vector size is fixed at layer construction).
    std::vector<InputSlot>::iterator BeginInputSlots() { return m_InputSlots.begin(); }
    std::vector<InputSlot>::iterator EndInputSlots() { return m_InputSlots.end(); }

    // Allow non-const access to output slots, but don't expose vector (vector size is fixed at layer construction).
    std::vector<OutputSlot>::iterator BeginOutputSlots() { return m_OutputSlots.begin(); }
    std::vector<OutputSlot>::iterator EndOutputSlots() { return m_OutputSlots.end(); }

    // Check whether the outputs of this layer don't have any connection
    bool IsOutputUnconnected()
    {
        unsigned int numConnections = 0;

        for (auto&& output : GetOutputSlots())
        {
            numConnections += output.GetNumConnections();
        }

        return (GetNumOutputSlots() > 0) && (numConnections == 0);
    }

    // Used for sorting
    void ResetPriority() const;
    LayerPriority GetPriority() const;

    LayerType GetType() const { return m_Type; }

    DataType GetDataType() const;

    Compute GetComputeDevice() const { return m_ComputeDevice; }
    void SetComputeDevice(Compute device) { m_ComputeDevice = device; }

    // Virtuals

    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const = 0;

    virtual void CreateTensorHandles(Graph& graph, const IWorkloadFactory& factory);

    /// Creates a dynamically-allocated copy of this layer
    /// @param graph The Graph into which this Layer is being cloned
    virtual Layer* Clone(Graph& graph) const = 0;

    virtual void ValidateTensorShapesFromInputs() = 0;

    /// Helper to serialize the layer parameters to string
    /// (currently used in DotSerializer and company)
    virtual void SerializeLayerParameters(ParameterStringifyFunction & fn) const {}

    // IConnectableLayer

    const char* GetName() const override { return m_LayerName.c_str(); }

    unsigned int GetNumInputSlots() const override { return static_cast<unsigned int>(m_InputSlots.size()); }
    unsigned int GetNumOutputSlots() const override { return static_cast<unsigned int>(m_OutputSlots.size()); }

    const InputSlot& GetInputSlot(unsigned int index) const override { return m_InputSlots.at(index); }
    InputSlot& GetInputSlot(unsigned int index) override { return  m_InputSlots.at(index); }
    const OutputSlot& GetOutputSlot(unsigned int index = 0) const override { return m_OutputSlots.at(index); }
    OutputSlot& GetOutputSlot(unsigned int index = 0) override { return m_OutputSlots.at(index); }

    void SetGuid(LayerGuid guid) { m_Guid = guid; }
    LayerGuid GetGuid() const final { return m_Guid; }

protected:
    // Graph needs access to the virtual destructor
    friend class Graph;
    virtual ~Layer() = default;

    template <typename QueueDescriptor>
    void CollectQueueDescriptorInputs(QueueDescriptor& descriptor, WorkloadInfo& info, const Graph& graph) const
    {
        WorkloadDataCollector dataCollector(descriptor.m_Inputs, info.m_InputTensorInfos);
        CollectWorkloadInputs(dataCollector, graph);
    }

    template <typename QueueDescriptor>
    void CollectQueueDescriptorOutputs(QueueDescriptor& descriptor, WorkloadInfo& info, const Graph& graph) const
    {
        WorkloadDataCollector dataCollector(descriptor.m_Outputs, info.m_OutputTensorInfos);
        CollectWorkloadOutputs(dataCollector, graph);
    }

    /// Helper function to reduce duplication in *Layer::CreateWorkload
    template <typename QueueDescriptor>
    WorkloadInfo PrepInfoAndDesc(QueueDescriptor& descriptor, const Graph& graph) const
    {
        WorkloadInfo info;
        CollectQueueDescriptorInputs(descriptor, info, graph);
        CollectQueueDescriptorOutputs(descriptor, info, graph);
        return info;
    }

    template <typename LayerType, typename ... Params>
    LayerType* CloneBase(Graph& graph, Params&& ... params) const;

private:
    void CollectWorkloadInputs(WorkloadDataCollector& dataCollector, const Graph& graph) const;
    void CollectWorkloadOutputs(WorkloadDataCollector& dataCollector, const Graph& graph) const;

protected:
    std::vector<OutputHandler> m_OutputHandlers;

private:
    const std::string m_LayerName;

    std::vector<InputSlot> m_InputSlots;
    std::vector<OutputSlot> m_OutputSlots;

    const LayerType m_Type;
    Compute m_ComputeDevice;

    /// Used for sorting
    mutable LayerPriority m_Priority = 0;
    mutable bool m_Visiting = false;

    LayerGuid m_Guid;
};

// A layer user-provided data can be bound to (e.g. inputs, outputs)
class BindableLayer : public Layer
{
public:
    BindableLayer(unsigned int numInputSlots,
        unsigned int numOutputSlots,
        LayerType type,
        const char* name,
        LayerBindingId id)
    : Layer(numInputSlots, numOutputSlots, type, name)
    , m_Id(id)
    {
    }

    LayerBindingId GetBindingId() const { return m_Id; };

protected:
    ~BindableLayer() = default;

private:
    LayerBindingId m_Id;
};

}
