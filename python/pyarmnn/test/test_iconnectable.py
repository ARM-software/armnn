# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
import pytest

import pyarmnn as ann


@pytest.fixture(scope="function")
def network():
    return ann.INetwork()


class TestIInputIOutputIConnectable:

    def test_input_slot(self, network):
        # Create input, addition & output layer
        input1 = network.AddInputLayer(0, "input1")
        input2 = network.AddInputLayer(1, "input2")
        add = network.AddAdditionLayer("addition")
        output = network.AddOutputLayer(0, "output")

        # Connect the input/output slots for each layer
        input1.GetOutputSlot(0).Connect(add.GetInputSlot(0))
        input2.GetOutputSlot(0).Connect(add.GetInputSlot(1))
        add.GetOutputSlot(0).Connect(output.GetInputSlot(0))

        # Check IInputSlot GetConnection()
        input_slot = add.GetInputSlot(0)
        input_slot_connection = input_slot.GetConnection()

        assert isinstance(input_slot_connection, ann.IOutputSlot)

        del input_slot_connection

        assert input_slot.GetConnection()
        assert isinstance(input_slot.GetConnection(), ann.IOutputSlot)

        del input_slot

        assert add.GetInputSlot(0)

    def test_output_slot(self, network):

        # Create input, addition & output layer
        input1 = network.AddInputLayer(0, "input1")
        input2 = network.AddInputLayer(1, "input2")
        add = network.AddAdditionLayer("addition")
        output = network.AddOutputLayer(0, "output")

        # Connect the input/output slots for each layer
        input1.GetOutputSlot(0).Connect(add.GetInputSlot(0))
        input2.GetOutputSlot(0).Connect(add.GetInputSlot(1))
        add.GetOutputSlot(0).Connect(output.GetInputSlot(0))

        # Check IInputSlot GetConnection()
        add_get_input_connection = add.GetInputSlot(0).GetConnection()
        output_get_input_connection = output.GetInputSlot(0).GetConnection()

        # Check IOutputSlot GetConnection()
        add_get_output_connect = add.GetOutputSlot(0).GetConnection(0)
        assert isinstance(add_get_output_connect.GetConnection(), ann.IOutputSlot)

        # Test IOutputSlot GetNumConnections() & CalculateIndexOnOwner()
        assert add_get_input_connection.GetNumConnections() == 1
        assert len(add_get_input_connection) == 1
        assert add_get_input_connection[0]
        assert add_get_input_connection.CalculateIndexOnOwner() == 0

        # Check GetOwningLayerGuid(). Check that it is different for add and output layer
        assert add_get_input_connection.GetOwningLayerGuid() != output_get_input_connection.GetOwningLayerGuid()

        # Set TensorInfo
        test_tensor_info = ann.TensorInfo(ann.TensorShape((2, 3)), ann.DataType_Float32)

        # Check IsTensorInfoSet()
        assert not add_get_input_connection.IsTensorInfoSet()
        add_get_input_connection.SetTensorInfo(test_tensor_info)
        assert add_get_input_connection.IsTensorInfoSet()

        # Check GetTensorInfo()
        output_tensor_info = add_get_input_connection.GetTensorInfo()
        assert 2 == output_tensor_info.GetNumDimensions()
        assert 6 == output_tensor_info.GetNumElements()

        # Check Disconnect()
        assert output_get_input_connection.GetNumConnections() == 1  # 1 connection to Outputslot0 from input1
        add.GetOutputSlot(0).Disconnect(output.GetInputSlot(0))  # disconnect add.OutputSlot0 from Output.InputSlot0
        assert output_get_input_connection.GetNumConnections() == 0

    def test_output_slot__out_of_range(self, network):
        # Create input layer to check output slot get item handling
        input1 = network.AddInputLayer(0, "input1")

        outputSlot = input1.GetOutputSlot(0)
        with pytest.raises(ValueError) as err:
                outputSlot[1]

        assert "Invalid index 1 provided" in str(err.value)

    def test_iconnectable_guid(self, network):

        # Check IConnectable GetGuid()
        # Note Guid can change based on which tests are run so
        # checking here that each layer does not have the same guid
        add_id = network.AddAdditionLayer().GetGuid()
        output_id = network.AddOutputLayer(0).GetGuid()
        assert add_id != output_id

    def test_iconnectable_layer_functions(self, network):

        # Create input, addition & output layer
        input1 = network.AddInputLayer(0, "input1")
        input2 = network.AddInputLayer(1, "input2")
        add = network.AddAdditionLayer("addition")
        output = network.AddOutputLayer(0, "output")

        # Check GetNumInputSlots(), GetName() & GetNumOutputSlots()
        assert input1.GetNumInputSlots() == 0
        assert input1.GetName() == "input1"
        assert input1.GetNumOutputSlots() == 1

        assert input2.GetNumInputSlots() == 0
        assert input2.GetName() == "input2"
        assert input2.GetNumOutputSlots() == 1

        assert add.GetNumInputSlots() == 2
        assert add.GetName() == "addition"
        assert add.GetNumOutputSlots() == 1

        assert output.GetNumInputSlots() == 1
        assert output.GetName() == "output"
        assert output.GetNumOutputSlots() == 0

        # Check GetOutputSlot()
        input1_get_output = input1.GetOutputSlot(0)
        assert input1_get_output.GetNumConnections() == 0
        assert len(input1_get_output) == 0

        # Check GetInputSlot()
        add_get_input = add.GetInputSlot(0)
        add_get_input.GetConnection()
        assert isinstance(add_get_input, ann.IInputSlot)
