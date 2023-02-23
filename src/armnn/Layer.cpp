//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "Layer.hpp"

#include "Graph.hpp"

#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadData.hpp>

#include <armnn/utility/NumericCast.hpp>

#include <armnnUtils/TensorUtils.hpp>

#include <client/include/IProfilingService.hpp>

#include <fmt/format.h>

#include <numeric>

namespace armnn
{

// Instantiate the static member variable
NullDescriptor Layer::m_NullDescriptor;

void AssertNumberOfInputSlots(Layer& layer)
{
    switch (layer.GetType())
    {
        case LayerType::Convolution2d:
        case LayerType::DepthwiseConvolution2d:
        case LayerType::FullyConnected:
        {
            ARMNN_ASSERT(layer.GetNumInputSlots() == 2 ||
                         layer.GetNumInputSlots() == 3);
            break;
        }
        default:
        {
            ARMNN_ASSERT(layer.GetNumInputSlots() == 1);
            break;
        }
    }
}

void InputSlot::Insert(Layer& layer)
{
    ARMNN_ASSERT(layer.GetNumOutputSlots() == 1);

    OutputSlot* const prevSlot = GetConnectedOutputSlot();

    if (prevSlot != nullptr)
    {
        // Disconnects parent from this.
        prevSlot->Disconnect(*this);

        AssertNumberOfInputSlots(layer);

        // Connects inserted layer to parent.
        int idx = prevSlot->Connect(layer.GetInputSlot(0));
        prevSlot->SetEdgeStrategy(armnn::numeric_cast<unsigned int>(idx), EdgeStrategy::Undefined);

        // Sets tensor info for inserted layer.
        const TensorInfo& tensorInfo = prevSlot->GetTensorInfo();
        layer.GetOutputHandler().SetTensorInfo(tensorInfo);
    }

    // Connects inserted layer to this.
    layer.GetOutputSlot(0).Connect(*this);
    layer.GetOutputSlot(0).SetEdgeStrategy(0, EdgeStrategy::Undefined);
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
    if (GetOwningLayer().GetShapeInferenceMethod() == ShapeInferenceMethod::InferAndValidate)
    {
        GetOwningLayer().ValidateTensorShapesFromInputs();
    }
    return GetOutputHandler().IsTensorInfoSet();
}

bool OutputSlot::ValidateTensorShape(const TensorShape& shape) const
{
    ARMNN_ASSERT_MSG(IsTensorInfoSet(), "TensorInfo must be set in order to validate the shape.");
    return shape == m_OutputHandler.GetTensorInfo().GetShape();
}

int OutputSlot::Connect(InputSlot& destination)
{
    destination.SetConnection(this);
    m_Connections.push_back(&destination);
    m_EdgeStrategies.push_back(EdgeStrategy::Undefined);
    return armnn::numeric_cast<int>(m_Connections.size() - 1);
}

void OutputSlot::Disconnect(InputSlot& slot)
{
    slot.SetConnection(nullptr);
    auto it = std::find(m_Connections.begin(), m_Connections.end(), &slot);

    if (it == m_Connections.end())
    {
        return;
    }

    auto idx = std::distance(m_Connections.begin(), it);
    m_Connections.erase(std::remove(m_Connections.begin(), m_Connections.end(), &slot), m_Connections.end());

    m_EdgeStrategies.erase(m_EdgeStrategies.begin() + idx);
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
        ARMNN_ASSERT_MSG(m_EdgeStrategies[0] == EdgeStrategy::Undefined,
            "Cannot move connections once memory strategies have be established.");

        InputSlot& connection = *GetConnection(0);
        Disconnect(connection);
        destination.Connect(connection);
        destination.GetOutputHandler().SetTensorInfo(GetOutputHandler().GetTensorInfo());
    }
}

unsigned int OutputSlot::CalculateIndexOnOwner() const
{
    for (unsigned int i = 0; i < GetOwningLayer().GetNumOutputSlots(); i++)
    {
        if (GetOwningLayer().GetOutputSlot(i) == (*this))
        {
            return i;
        }
    }
    ARMNN_ASSERT_MSG(false, "Did not find slot on owner.");
    return 0; // Error
}

bool OutputSlot::operator==(const OutputSlot& other) const
{
    bool isSame = other.GetNumConnections() == GetNumConnections();
    if (!isSame)
    {
        return false;
    }

    for (unsigned int i = 0; i < GetNumConnections(); i++)
    {
        isSame &= other.GetConnection(i) == GetConnection(i);
    }
    return isSame;
}

void OutputSlot::ValidateConnectionIndex(unsigned int index) const
{
    if (armnn::numeric_cast<std::size_t>(index) >= m_Connections.size())
    {
        throw InvalidArgumentException((fmt::format("GetConnection: Invalid index {} provided", index)));
    }
}

LayerGuid OutputSlot::GetOwningLayerGuid() const
{
    return GetOwningLayer().GetGuid();
}

void OutputSlot::SetTensorHandleFactory(const ITensorHandleFactory::FactoryId& id)
{
    m_TensorHandleFactoryId = id;
}

ITensorHandleFactory::FactoryId OutputSlot::GetTensorHandleFactoryId() const
{
    return m_TensorHandleFactoryId;
}

void OutputSlot::SetEdgeStrategy(unsigned int connectionIndex, EdgeStrategy strategy)
{
    m_EdgeStrategies[connectionIndex] = strategy;
}

EdgeStrategy OutputSlot::GetEdgeStrategyForConnection(unsigned int connectionIdx) const
{
    return m_EdgeStrategies[connectionIdx];
}

Layer::Layer(unsigned int numInputSlots,
             unsigned int numOutputSlots,
             LayerType type,
             DataLayout layout,
             const char* name)
: m_OutputHandlers(numOutputSlots)
, m_ShapeInferenceMethod(ShapeInferenceMethod::ValidateOnly)
, m_LayerName(name ? name : "")
, m_Type(type)
, m_BackendId()
, m_BackendHint(EmptyOptional())
, m_Guid(arm::pipe::IProfilingService::GetNextGuid())
{
    IgnoreUnused(layout);
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

Layer::Layer(unsigned int numInputSlots,
             unsigned int numOutputSlots,
             LayerType type,
             const char* name)
: Layer(numInputSlots, numOutputSlots, type, DataLayout::NCHW, name)
{
}

void Layer::CollectWorkloadInputs(WorkloadDataCollector& dataCollector) const
{
    for (auto&& inputSlot : GetInputSlots())
    {
        // The graph must be well-formed at this point.
        ARMNN_ASSERT(inputSlot.GetConnection());
        const OutputHandler& outputHandler = inputSlot.GetConnectedOutputSlot()->GetOutputHandler();
        dataCollector.Push(outputHandler.GetData(), outputHandler.GetTensorInfo());
    }
}

void Layer::CollectWorkloadOutputs(WorkloadDataCollector& dataCollector) const
{
    for (auto&& outputHandler : m_OutputHandlers)
    {
        outputHandler.CollectWorkloadOutputs(dataCollector);
    }
}

void Layer::SetAdditionalInfo(QueueDescriptor& descriptor) const
{
    descriptor.m_AdditionalInfoObject = m_AdditionalInfoObject.get();
}

void Layer::CreateTensorHandles(const TensorHandleFactoryRegistry& registry,
                                const IWorkloadFactory& workloadFactory,
                                const bool IsMemoryManaged)
{
    for (unsigned int idx=0; idx < GetNumOutputSlots(); idx++)
    {

        OutputSlot& slot = GetOutputSlot(idx);
        ITensorHandleFactory::FactoryId factoryId = slot.GetTensorHandleFactoryId();

        OutputHandler& handler = GetOutputHandler(idx);
        if (factoryId == ITensorHandleFactory::LegacyFactoryId)
        {
            handler.CreateTensorHandles(workloadFactory, IsMemoryManaged);
        }
        else
        {
            ITensorHandleFactory* handleFactory;
            handleFactory = registry.GetFactory(factoryId);
            ARMNN_ASSERT(handleFactory);
            handler.CreateTensorHandles(*handleFactory, IsMemoryManaged);
        }
    }
}

void Layer::ReleaseConstantData()
{
    // Now free up the static data.
    OperateOnConstantTensors([](std::shared_ptr<ConstTensorHandle>& handle)
                                 {
                                     handle.reset();
                                 });
}

DataType Layer::GetDataType() const
{
    if (GetNumInputSlots() > 0) // Ignore the input layer.
    {
        return GetInputSlot(0).GetConnection()->GetTensorInfo().GetDataType();
    }
    return GetOutputSlot(0).GetTensorInfo().GetDataType();
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
                const OutputSlot *outputSlot = slot.GetConnectedOutputSlot();
                if (outputSlot)
                {
                    const Layer& input = outputSlot->GetOwningLayer();
                    return std::max(prio, input.GetPriority());
                }
                else
                {
                    // unconnected input slot
                    return prio;
                }
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

void Layer::VerifyLayerConnections(unsigned int expectedConnections, const CheckLocation& location) const
{
    ARMNN_ASSERT(GetNumInputSlots() == expectedConnections);

    for (unsigned int i=0; i<expectedConnections; ++i)
    {
        if (GetInputSlot(i).GetConnection() == nullptr)
        {
            throw LayerValidationException(
                    fmt::format("Input connection #{0} must be connected "
                                "for {1} layer {2} {3}",
                                i,
                                GetLayerTypeAsCString(this->GetType()),
                                GetNameStr(),
                                location.AsString()));
        }
    }
}

std::vector<TensorShape> Layer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(GetNumInputSlots() != 0);
    ARMNN_ASSERT(GetNumOutputSlots() != 0);

    // By default we return what we got, meaning the output shape(s) are the same as the input(s).
    // This only works if the number of inputs and outputs are the same. Since we are in the Layer
    // base class, this means the implementation needs to be overridden in the specific layers for
    // the other cases. So the missing implementation justifies the UnimplementedException.

    if (GetNumInputSlots() != GetNumOutputSlots())
    {
        throw UnimplementedException(
                fmt::format("Default implementation for InferOutputShapes can only be used for "
                            "layers with the same number of input and output slots. This doesn't "
                            "hold for {0} layer {1} (#inputs={2} #outputs={3}) {4}",
                            GetLayerTypeAsCString(this->GetType()),
                            GetNameStr(),
                            GetNumInputSlots(),
                            GetNumOutputSlots(),
                            CHECK_LOCATION().AsString()));
    }
    return inputShapes;
}

void Layer::ValidateAndCopyShape(const TensorShape& outputShape,
                                 const TensorShape& inferredShape,
                                 const ShapeInferenceMethod shapeInferenceMethod,
                                 const std::string& layerName,
                                 const unsigned int outputSlotIndex)
{
    if (shapeInferenceMethod == ShapeInferenceMethod::ValidateOnly)
    {
        if (m_AllowExpandedDims)
        {
            std::vector<unsigned int> outputDims = armnnUtils::SqueezeDims(outputShape);
            std::vector<unsigned int> inferredDims = armnnUtils::SqueezeDims(inferredShape);

            if (outputDims.size() != inferredDims.size())
            {
                std::stringstream ss;
                ss << layerName << ": TensorShape set on OutputSlot[" << outputSlotIndex <<
                   "] does not match the inferred shape. ";
                ss << outputShape << " != " << inferredShape;
                throw LayerValidationException(ss.str());
            }
            for (unsigned int i = 0; i < outputDims.size(); ++i)
            {
                if (outputDims[i] != inferredDims[i])
                {
                    std::stringstream ss;
                    ss << layerName << ": TensorShape set on OutputSlot[" << outputSlotIndex <<
                       "] does not match the inferred shape at dimension index [";
                    ss << i << "] " << outputShape << " != " << inferredShape;
                    throw LayerValidationException(ss.str());
                }
            }
            return;
        }
        else
        {
            ConditionalThrowIfNotEqual<LayerValidationException>(
                    layerName + ": TensorShape set on OutputSlot[0] does not match the inferred shape.",
                    outputShape,
                    inferredShape);
            return;
        }
    }

    if (outputShape.GetDimensionality() == Dimensionality::Specified)
    {
        for (unsigned int i = 0; i < outputShape.GetNumDimensions(); ++i)
        {
            if (outputShape.GetDimensionSpecificity(i) && outputShape[i] != inferredShape[i])
            {
                std::stringstream ss;
                ss << layerName << ": TensorShape set on OutputSlot[" << outputSlotIndex <<
                "] does not match the inferred shape at dimension index [";
                ss << i << "] " << outputShape << " != " << inferredShape;
                throw LayerValidationException(ss.str());
            }
        }
    }

    TensorInfo info = GetOutputSlot(outputSlotIndex).GetTensorInfo();

    armnn::TensorInfo inferredTensorInfo(inferredShape,
                                         info.GetDataType(),
                                         info.GetQuantizationScale(),
                                         info.GetQuantizationOffset());

    GetOutputSlot(outputSlotIndex).SetTensorInfo(inferredTensorInfo);
}

void Layer::VerifyShapeInferenceType(const TensorShape& outputShape, ShapeInferenceMethod shapeInferenceMethod)
{
    if (shapeInferenceMethod == ShapeInferenceMethod::ValidateOnly)
    {
        ConditionalThrow<LayerValidationException>(
                outputShape.GetDimensionality() != Dimensionality::NotSpecified,
                "Dimensionality can not be NotSpecified while using ShapeInferenceMethod::ValidateOnly");

        ConditionalThrow<LayerValidationException>(
                outputShape.AreAllDimensionsSpecified(),
                "Unspecified dimension while using ShapeInferenceMethod::ValidateOnly");
    }
}

void Layer::SerializeLayerParameters(ParameterStringifyFunction& fn) const
{
    std::string guid = std::to_string(m_Guid);
    std::string layerType = GetLayerTypeAsCString(m_Type);
    std::string backendId = std::string(m_BackendId);
    if (!(guid.compare("") == 0) && !guid.empty())
    {
        fn("Guid", guid);
    }
    if(!(m_LayerName.compare("") == 0) && !m_LayerName.empty())
    {
        fn("LayerName",m_LayerName);
    }
    if(!(layerType.compare("") == 0) && !layerType.empty())
    {
        fn("LayerType",layerType);
    }
    if(!(backendId.compare("") == 0) && !backendId.empty())
    {
        fn("BackendID",backendId);
    }
    std::shared_ptr<ActivationDescriptor>
            activationDescPtr = GetAdditionalInformation<ActivationDescriptor>();

    if (activationDescPtr)
    {
        StringifyLayerParameters<ActivationDescriptor>::Serialize(fn, *activationDescPtr.get());
    }
}

// default implementation of ExecuteStrategy
void Layer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, BaseDescriptor(), {}, GetName());
}

Layer::ConstantTensors Layer::GetConstantTensorsByRef()
{
    const Layer *constThis = const_cast<const Layer*>(this);
    ConstantTensors res;

    ImmutableConstantTensors immutableData = constThis->GetConstantTensorsByRef();
    for (auto i : immutableData)
    {
        res.push_back(const_cast<std::shared_ptr<ConstTensorHandle>&>(i.get()));
    }
    return res;
}

const IConnectableLayer& OutputSlot::GetOwningIConnectableLayer() const
{
    return m_OwningLayer;
}

IConnectableLayer& OutputSlot::GetOwningIConnectableLayer()
{
    return m_OwningLayer;
}

const IConnectableLayer& InputSlot::GetOwningIConnectableLayer() const
{
    return m_OwningLayer;
}

IConnectableLayer& InputSlot::GetOwningIConnectableLayer()
{
    return m_OwningLayer;
}

} // namespace armnn
