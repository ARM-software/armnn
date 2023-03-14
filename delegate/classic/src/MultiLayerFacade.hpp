//
// Copyright © 2021,2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

// NOTE: the MultiLayerFacade class is a utility class which makes a chain
//       of operators look like a single IConnectableLayer with the first
//       layer in the chain supplying the input slots and the last supplying
//       the output slots. It enables us, for example, to simulate a
//       Tensorflow Lite FloorDiv operator by chaining a Div layer followed
//       by a Floor layer and pass them as a single unit to the code that
//       connects up the graph as the delegate proceeds to build up the
//       Arm NN subgraphs.
//

#include <common/include/ProfilingGuid.hpp>
#include <armnn/INetwork.hpp>

namespace armnnDelegate
{

class MultiLayerFacade : public armnn::IConnectableLayer
{
public:
    MultiLayerFacade() :
        m_FirstLayer(nullptr), m_LastLayer(nullptr) {}

    MultiLayerFacade(armnn::IConnectableLayer* firstLayer, armnn::IConnectableLayer* lastLayer) :
        m_FirstLayer(firstLayer), m_LastLayer(lastLayer) {}

    MultiLayerFacade(const MultiLayerFacade& obj) :
        m_FirstLayer(obj.m_FirstLayer), m_LastLayer(obj.m_LastLayer) {}

    ~MultiLayerFacade() {} // we don't own the pointers

    MultiLayerFacade& operator=(const MultiLayerFacade& obj)
    {
        m_FirstLayer = obj.m_FirstLayer;
        m_LastLayer = obj.m_LastLayer;
        return *this;
    }

    void AssignValues(armnn::IConnectableLayer* firstLayer, armnn::IConnectableLayer* lastLayer)
    {
        m_FirstLayer = firstLayer;
        m_LastLayer = lastLayer;
    }

    virtual const char* GetName() const override
    {
        return m_FirstLayer->GetName();
    }

    virtual unsigned int GetNumInputSlots() const override
    {
        return m_FirstLayer->GetNumInputSlots();
    }

    virtual unsigned int GetNumOutputSlots() const override
    {
        return m_LastLayer->GetNumOutputSlots();
    }

    virtual const armnn::IInputSlot& GetInputSlot(unsigned int index) const override
    {
        return m_FirstLayer->GetInputSlot(index);
    }

    virtual armnn::IInputSlot& GetInputSlot(unsigned int index) override
    {
        return m_FirstLayer->GetInputSlot(index);
    }

    virtual const armnn::IOutputSlot& GetOutputSlot(unsigned int index) const override
    {
        return m_LastLayer->GetOutputSlot(index);
    }

    virtual armnn::IOutputSlot& GetOutputSlot(unsigned int index) override
    {
        return m_LastLayer->GetOutputSlot(index);
    }

    virtual std::vector<armnn::TensorShape> InferOutputShapes(
        const std::vector<armnn::TensorShape>& inputShapes) const override
    {
        // NOTE: do not expect this function to be used. Likely that if it is it might need to be overridden
        //       for particular sequences of operators.
        return m_FirstLayer->InferOutputShapes(inputShapes);
    }

    virtual LayerGuid GetGuid() const override
    {
        return m_FirstLayer->GetGuid();
    }

    virtual void ExecuteStrategy(armnn::IStrategy& strategy) const override
    {
        // Do not expect this function to be used so not providing an implementation
        // if an implementation is required and the chain contains more than two operators
        // would have to provide a way to record the intermediate layers so they could be
        // visited... the same applies to the BackendSelectionHint
        // below.
    }

    virtual void BackendSelectionHint(armnn::Optional<armnn::BackendId> backend) override
    {
        // Do not expect this function to be used so not providing an implementation
    }

    virtual armnn::LayerType GetType() const override
    {
        return m_FirstLayer->GetType();
    }

    virtual const armnn::BaseDescriptor& GetParameters() const override { return m_NullDescriptor; }

    void SetBackendId(const armnn::BackendId& id) override {}

protected:
    /// Retrieve the handles to the constant values stored by the layer.
    /// @return A vector of the constant tensors stored by this layer.
    ConstantTensors GetConstantTensorsByRef() override { return {}; }
    ImmutableConstantTensors GetConstantTensorsByRef() const override { return {}; }

private:
    armnn::IConnectableLayer* m_FirstLayer;
    armnn::IConnectableLayer* m_LastLayer;

    // to satisfy the GetParameters method need to hand back a NullDescriptor
    armnn::NullDescriptor m_NullDescriptor;
};

} // namespace armnnDelegate
