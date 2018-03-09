//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "Layer.hpp"

#include "Graph.hpp"
#include "backends/WorkloadData.hpp"

#include <boost/cast.hpp>
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>

#include <numeric>

namespace armnn
{

void InputSlot::Insert(Layer& layer)
{
    BOOST_ASSERT(layer.GetNumInputSlots() <= 1);
    BOOST_ASSERT(layer.GetNumOutputSlots() == 1);

    OutputSlot* const prevSlot = GetConnectedOutputSlot();

    if (prevSlot != nullptr)
    {
        // Disconnect parent from this
        prevSlot->Disconnect(*this);

        // Connect inserted layer to parent
        BOOST_ASSERT(layer.GetNumInputSlots() == 1);
        prevSlot->Connect(layer.GetInputSlot(0));

        // Set tensor info for inserted layer
        const TensorInfo& tensorInfo = prevSlot->GetTensorInfo();
        layer.GetOutputHandler().SetTensorInfo(tensorInfo);
    }

    // Connect inserted layer to this
    layer.GetOutputSlot(0).Connect(*this);
}

const InputSlot* OutputSlot::GetConnection(unsigned int index) const
{
    ValidateConnectionIndex(index);
    return m_Connections[index];
}

InputSlot* OutputSlot::GetConnection(unsigned int index)
{
    ValidateConnectionIndex(index);
    return m_Connections[index];
}

void OutputSlot::SetTensorInfo(const TensorInfo& tensorInfo)
{
    GetOutputHandler().SetTensorInfo(tensorInfo);
}

const TensorInfo& OutputSlot::GetTensorInfo() const
{
    return GetOutputHandler().GetTensorInfo();
}

bool OutputSlot::IsTensorInfoSet() const
{
    return GetOutputHandler().IsTensorInfoSet();
}

bool OutputSlot::ValidateTensorShape(const TensorShape& shape) const
{
    BOOST_ASSERT_MSG(IsTensorInfoSet(), "TensorInfo must be set in order to validate the shape.");
    return shape == m_OutputHandler.GetTensorInfo().GetShape();
}

int OutputSlot::Connect(InputSlot& destination)
{
    destination.SetConnection(this);
    m_Connections.push_back(&destination);
    return boost::numeric_cast<int>(m_Connections.size() - 1);
}

void OutputSlot::Disconnect(InputSlot& slot)
{
    slot.SetConnection(nullptr);
    m_Connections.erase(std::remove(m_Connections.begin(), m_Connections.end(), &slot), m_Connections.end());
}

void OutputSlot::DisconnectAll()
{
    while (GetNumConnections() > 0)
    {
        InputSlot& connection = *GetConnection(0);
        Disconnect(connection);
    }
}

void OutputSlot::MoveAllConnections(OutputSlot& destination)
{
    while (GetNumConnections() > 0)
    {
        InputSlot& connection = *GetConnection(0);
        Disconnect(connection);
        destination.Connect(connection);
    }
}

void OutputSlot::ValidateConnectionIndex(unsigned int index) const
{
    if (boost::numeric_cast<std::size_t>(index) >= m_Connections.size())
    {
        throw InvalidArgumentException(
            boost::str(boost::format("GetConnection: Invalid index %1% provided") % index));
    }
}

Layer::Layer(unsigned int numInputSlots, unsigned int numOutputSlots, LayerType type, const char* name)
: m_OutputHandlers(numOutputSlots)
, m_LayerName(name ? name : "")
, m_Type(type)
, m_ComputeDevice(Compute::Undefined)
{
    m_InputSlots.reserve(numInputSlots);
    for (unsigned int i = 0; i < numInputSlots; ++i)
    {
        m_InputSlots.emplace_back(*this, i);
    }

    m_OutputSlots.reserve(numOutputSlots);
    for (unsigned int i = 0; i < numOutputSlots; ++i)
    {
        m_OutputSlots.emplace_back(*this, m_OutputHandlers[i]);
    }
}

void Layer::CollectWorkloadInputs(WorkloadDataCollector& dataCollector, const Graph& graph) const
{
    for (auto&& inputSlot : GetInputSlots())
    {
        // The graph must be well-formed at this point
        BOOST_ASSERT(inputSlot.GetConnection());
        const OutputHandler& outputHandler = inputSlot.GetConnectedOutputSlot()->GetOutputHandler();
        dataCollector.Push(outputHandler.GetData(), outputHandler.GetTensorInfo());
    }
}

void Layer::CollectWorkloadOutputs(WorkloadDataCollector& dataCollector, const Graph& graph) const
{
    for (auto&& outputHandler : m_OutputHandlers)
    {
        outputHandler.CollectWorkloadOutputs(dataCollector);
    }
}

void Layer::CreateTensorHandles(Graph& graph, const IWorkloadFactory& factory)
{
    for (auto&& outputHandler : m_OutputHandlers)
    {
        outputHandler.CreateTensorHandles(factory);
    }
}

DataType Layer::GetDataType() const
{
    if (GetNumInputSlots() > 0) // Ignore the input layer
    {
        return GetInputSlot(0).GetConnection()->GetTensorInfo().GetDataType();
    }
    return DataType::Float32;
}

void Layer::ResetPriority() const
{
    m_Priority = 0;
    m_Visiting = false;
}

LayerPriority Layer::GetPriority() const
{
    constexpr LayerPriority inputPrio = std::numeric_limits<LayerPriority>::lowest();
    constexpr LayerPriority outputPrio = std::numeric_limits<LayerPriority>::max();

    if (GetType() == LayerType::Input)
    {
        m_Priority = inputPrio;
    }
    else if (GetType() == LayerType::Output)
    {
        m_Priority = outputPrio;
    }
    else if (m_Priority == 0)
    {
        if (m_Visiting)
        {
            throw GraphValidationException("Graph has circular dependencies: cannot walk");
        }

        auto maxPrio = [](const LayerPriority prio, const InputSlot& slot) -> LayerPriority
            {
                const Layer& input = slot.GetConnectedOutputSlot()->GetOwningLayer();
                return std::max(prio, input.GetPriority());
            };

        m_Visiting = true;
        LayerPriority parentPrio = std::accumulate(GetInputSlots().cbegin(), GetInputSlots().cend(), 0U, maxPrio);
        m_Visiting = false;

        if (parentPrio >= outputPrio)
        {
            throw GraphValidationException("Graph has too many edges");
        }

        m_Priority = parentPrio + 1U;
    }

    return m_Priority;
}

} // namespace armnn
