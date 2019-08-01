//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerFwd.hpp"

#include <backendsCommon/ITensorHandleFactory.hpp>
#include <backendsCommon/OutputHandler.hpp>
#include <backendsCommon/TensorHandleFactoryRegistry.hpp>
#include <backendsCommon/WorkloadDataCollector.hpp>
#include <backendsCommon/WorkloadInfo.hpp>
#include "InternalTypes.hpp"
#include "SerializeLayerParameters.hpp"

#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/INetwork.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <functional>
#include <list>

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

    /// Links the slot to an output slot or breaks an existing link if passing nullptr.
    void SetConnection(OutputSlot* source)
    {
        if (m_Connection != nullptr && source != nullptr)
        {
            throw InvalidArgumentException("Tried to connect an output slot to an input slot, "
                "but the latter already has a connection");
        }
        m_Connection = source;
    }

    // Inserts single-output existing layer at this point in the graph.
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
    , m_TensorHandleFactoryId(ITensorHandleFactory::LegacyFactoryId)
    {}

    OutputSlot(const OutputSlot&) = delete;
    OutputSlot& operator=(const OutputSlot&) = delete;

    OutputSlot(OutputSlot&&) = default;
    OutputSlot& operator=(OutputSlot&&) = default;

    ~OutputSlot()
    {
        try
        {
            // Coverity fix: DisconnectAll() may throw uncaught exceptions.
            DisconnectAll();
        }
        catch (const std::exception& e)
        {
            // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
            // exception of type std::length_error.
            // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
            std::cerr << "WARNING: An error has occurred when disconnecting all output slots: "
                      << e.what() << std::endl;
        }
    }

    Layer& GetOwningLayer() const { return m_OwningLayer; }

    LayerGuid GetOwningLayerGuid() const override;

    const OutputHandler& GetOutputHandler() const { return m_OutputHandler; }
    OutputHandler& GetOutputHandler() { return m_OutputHandler; }

    int Connect(InputSlot& destination);
    void Disconnect(InputSlot& slot);

    const std::vector<InputSlot*>& GetConnections() const { return m_Connections; }
    const std::vector<EdgeStrategy>& GetEdgeStrategies() const { return m_EdgeStrategies; }

    bool ValidateTensorShape(const TensorShape& shape) const;

    // Disconnect all conections.
    void DisconnectAll();

    /// Moves all connections to another OutputSlot.
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

    unsigned int CalculateIndexOnOwner() const override;

    bool operator==(const OutputSlot& other) const;

    void SetTensorHandleFactory(const ITensorHandleFactory::FactoryId& id);
    ITensorHandleFactory::FactoryId GetTensorHandleFactoryId() const;

    void SetEdgeStrategy(unsigned int connectionIndex, EdgeStrategy strategy);
    EdgeStrategy GetEdgeStrategyForConnection(unsigned int connectionIdx) const;

private:
    void ValidateConnectionIndex(unsigned int index) const;

    Layer& m_OwningLayer;
    OutputHandler& m_OutputHandler;
    std::vector<InputSlot*> m_Connections;

    ITensorHandleFactory::FactoryId m_TensorHandleFactoryId;
    std::vector<EdgeStrategy> m_EdgeStrategies;
};

// InputSlot inlines that need OutputSlot declaration.

inline InputSlot::~InputSlot()
{
    if (m_Connection != nullptr)
    {
        try
        {
            // Coverity fix: Disconnect() may throw uncaught exceptions.
            m_Connection->Disconnect(*this);
        }
        catch (const std::exception& e)
        {
            // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
            // exception of type std::length_error.
            // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
            std::cerr << "WARNING: An error has occurred when disconnecting an input slot: "
                      << e.what() << std::endl;
        }
    }
}

inline const IOutputSlot* InputSlot::GetConnection() const { return GetConnectedOutputSlot(); }
inline IOutputSlot* InputSlot::GetConnection() { return GetConnectedOutputSlot(); }


class ScopedCpuTensorHandle;

// Base layer class

using LayerPriority = unsigned int;

class Layer : public IConnectableLayer
{
public:
    /// @param name - Optional name for the layer (may be nullptr).
    Layer(unsigned int numInputSlots, unsigned int numOutputSlots, LayerType type, const char* name);
    Layer(unsigned int numInputSlots, unsigned int numOutputSlots, LayerType type, DataLayout layout, const char* name);

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

    // Allows non-const access to input slots, but don't expose vector (vector size is fixed at layer construction).
    std::vector<InputSlot>::iterator BeginInputSlots() { return m_InputSlots.begin(); }
    std::vector<InputSlot>::iterator EndInputSlots() { return m_InputSlots.end(); }

    // Allows non-const access to output slots, but don't expose vector (vector size is fixed at layer construction).
    std::vector<OutputSlot>::iterator BeginOutputSlots() { return m_OutputSlots.begin(); }
    std::vector<OutputSlot>::iterator EndOutputSlots() { return m_OutputSlots.end(); }

    // Checks whether the outputs of this layer don't have any connection.
    bool IsOutputUnconnected()
    {
        unsigned int numConnections = 0;

        for (auto&& output : GetOutputSlots())
        {
            numConnections += output.GetNumConnections();
        }

        return (GetNumOutputSlots() > 0) && (numConnections == 0);
    }

    // Used for sorting.
    void ResetPriority() const;
    LayerPriority GetPriority() const;

    LayerType GetType() const { return m_Type; }

    DataType GetDataType() const;

    const BackendId& GetBackendId() const { return m_BackendId; }
    void SetBackendId(const BackendId& id) { m_BackendId = id; }

    // Virtuals

    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const = 0;

    virtual void CreateTensorHandles(const TensorHandleFactoryRegistry& registry, const IWorkloadFactory& factory);

    /// Creates a dynamically-allocated copy of this layer.
    /// @param graph - The Graph into which this Layer is being cloned.
    virtual Layer* Clone(Graph& graph) const = 0;

    void VerifyLayerConnections(unsigned int expectedConnections, const CheckLocation& location) const;

    virtual void ValidateTensorShapesFromInputs() = 0;

    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    /// Helper to serialize the layer parameters to string.
    /// (currently used in DotSerializer and company).
    virtual void SerializeLayerParameters(ParameterStringifyFunction &) const {}

    // Free up the constant source data
    virtual void ReleaseConstantData();

    template<typename Op>
    void OperateOnConstantTensors(Op op)
    {
        for (auto constant : GetConstantTensorsByRef())
        {
            if (constant.get())
            {
                op(constant);
            }
        }
    };

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

    void AddRelatedLayerName(const std::string layerName) { m_RelatedLayerNames.emplace_back(layerName); }

    const std::list<std::string>& GetRelatedLayerNames() { return m_RelatedLayerNames; }

    virtual void Reparent(Graph& dest, std::list<Layer*>::const_iterator iterator) = 0;
protected:
    // Graph needs access to the virtual destructor.
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

    /// Helper function to reduce duplication in *Layer::CreateWorkload.
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

    // Retrieve the Handles to the constants
    using ConstantTensors = std::vector<std::reference_wrapper<std::unique_ptr<ScopedCpuTensorHandle>>>;
    virtual ConstantTensors GetConstantTensorsByRef() {return ConstantTensors(); };

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
    BackendId m_BackendId;

    /// Used for sorting.
    mutable LayerPriority m_Priority = 0;
    mutable bool m_Visiting = false;

    LayerGuid m_Guid;

    std::list<std::string> m_RelatedLayerNames;
};

// A layer user-provided data can be bound to (e.g. inputs, outputs).
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
