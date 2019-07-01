//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TestUtils.hpp"

#include <armnn/ArmNN.hpp>
#include <Graph.hpp>
#include <Optimizer.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <FloatingPointConverter.hpp>

#include <boost/test/unit_test.hpp>

using namespace armnn;

namespace
{
template <typename LayerT>
bool IsLayerOfType(const armnn::Layer* const layer)
{
    return (layer->GetType() == armnn::LayerEnumOf<LayerT>());
}

bool CheckSequence(const armnn::Graph::ConstIterator first, const armnn::Graph::ConstIterator last)
{
    return (first == last);
}

/// Checks each unary function in Us evaluates true for each correspondent layer in the sequence [first, last).
template <typename U, typename... Us>
bool CheckSequence(const armnn::Graph::ConstIterator first,
                   const armnn::Graph::ConstIterator last,
                   U&& u,
                   Us&&... us)
{
    return u(*first) && CheckSequence(std::next(first), last, us...);
}

template <typename LayerT>
bool CheckRelatedLayers(armnn::Graph& graph, const std::list<std::string>& testRelatedLayers)
{
    for (auto& layer : graph)
    {
        if (layer->GetType() == armnn::LayerEnumOf<LayerT>())
        {
            auto& relatedLayers = layer->GetRelatedLayerNames();
            if(!std::equal(relatedLayers.begin(), relatedLayers.end(),
                           testRelatedLayers.begin(), testRelatedLayers.end()))
            {
                return false;
            }
        }
    }

    return true;
}

void CreateLSTMLayerHelper(Graph &graph, bool CifgEnabled)
{
    LstmDescriptor layerDesc;
    layerDesc.m_ActivationFunc = 4;
    layerDesc.m_ClippingThresCell = 0.2f;
    layerDesc.m_ClippingThresProj = 0.4f;
    layerDesc.m_CifgEnabled = CifgEnabled;
    layerDesc.m_PeepholeEnabled = false;
    layerDesc.m_ProjectionEnabled = false;

    LstmLayer* const layer = graph.AddLayer<LstmLayer>(layerDesc, "layer");
    unsigned int batchSize = 3;
    unsigned int inputSize = 2;
    unsigned int numUnits = 4;
    unsigned int outputSize = 4;

    layer->m_BasicParameters.m_InputToForgetWeights = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits, inputSize }, DataType::Float32));
    layer->m_BasicParameters.m_InputToCellWeights = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits, inputSize }, DataType::Float32));
    layer->m_BasicParameters.m_InputToOutputWeights = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits, inputSize }, DataType::Float32));
    layer->m_BasicParameters.m_RecurrentToForgetWeights = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits, outputSize }, DataType::Float32));
    layer->m_BasicParameters.m_RecurrentToCellWeights = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits, outputSize }, DataType::Float32));
    layer->m_BasicParameters.m_RecurrentToOutputWeights = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits, outputSize }, DataType::Float32));
    layer->m_BasicParameters.m_ForgetGateBias = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits }, DataType::Float32));
    layer->m_BasicParameters.m_CellBias = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits }, DataType::Float32));
    layer->m_BasicParameters.m_OutputGateBias = std::make_unique<ScopedCpuTensorHandle>
            (TensorInfo({ numUnits }, DataType::Float32));

    layer->m_BasicParameters.m_InputToForgetWeights->Allocate();
    layer->m_BasicParameters.m_InputToCellWeights->Allocate();
    layer->m_BasicParameters.m_InputToOutputWeights->Allocate();
    layer->m_BasicParameters.m_RecurrentToForgetWeights->Allocate();
    layer->m_BasicParameters.m_RecurrentToCellWeights->Allocate();
    layer->m_BasicParameters.m_RecurrentToOutputWeights->Allocate();
    layer->m_BasicParameters.m_ForgetGateBias->Allocate();
    layer->m_BasicParameters.m_CellBias->Allocate();
    layer->m_BasicParameters.m_OutputGateBias->Allocate();

    if (!layerDesc.m_CifgEnabled)
    {
        layer->m_CifgParameters.m_InputToInputWeights = std::make_unique<ScopedCpuTensorHandle>
                (TensorInfo({ numUnits, inputSize }, DataType::Float32));
        layer->m_CifgParameters.m_RecurrentToInputWeights = std::make_unique<ScopedCpuTensorHandle>
                (TensorInfo({ numUnits, outputSize }, DataType::Float32));
        layer->m_CifgParameters.m_CellToInputWeights = std::make_unique<ScopedCpuTensorHandle>
                (TensorInfo({ numUnits }, DataType::Float32));
        layer->m_CifgParameters.m_InputGateBias = std::make_unique<ScopedCpuTensorHandle>
                (TensorInfo({ numUnits }, DataType::Float32));
        layer->m_CifgParameters.m_InputToInputWeights->Allocate();
        layer->m_CifgParameters.m_RecurrentToInputWeights->Allocate();
        layer->m_CifgParameters.m_CellToInputWeights->Allocate();
        layer->m_CifgParameters.m_InputGateBias->Allocate();
    }

    if (layerDesc.m_ProjectionEnabled)
    {
        layer->m_ProjectionParameters.m_ProjectionWeights = std::make_unique<ScopedCpuTensorHandle>
                (TensorInfo({ outputSize, numUnits }, DataType::Float32));
        layer->m_ProjectionParameters.m_ProjectionBias = std::make_unique<ScopedCpuTensorHandle>
                (TensorInfo({ outputSize }, DataType::Float32));
        layer->m_ProjectionParameters.m_ProjectionWeights->Allocate();
        layer->m_ProjectionParameters.m_ProjectionBias->Allocate();
    }

    if (layerDesc.m_PeepholeEnabled)
    {
        layer->m_PeepholeParameters.m_CellToForgetWeights = std::make_unique<ScopedCpuTensorHandle>
                (TensorInfo({ numUnits }, DataType::Float32));
        layer->m_PeepholeParameters.m_CellToOutputWeights = std::make_unique<ScopedCpuTensorHandle>
                (TensorInfo({ numUnits }, DataType::Float32));
        layer->m_PeepholeParameters.m_CellToForgetWeights->Allocate();
        layer->m_PeepholeParameters.m_CellToOutputWeights->Allocate();
    }

    // create input and output layers
    Layer* const input = graph.AddLayer<InputLayer>(0, "input");
    Layer* const outputStateIn = graph.AddLayer<InputLayer>(1, "outputStateIn");
    Layer* const cellStateIn = graph.AddLayer<InputLayer>(2, "cellStateIn");
    Layer* const scratchBuffer = graph.AddLayer<OutputLayer>(0, "scratchBuffer");
    Layer* const outputStateOut = graph.AddLayer<OutputLayer>(1, "outputStateOut");
    Layer* const cellStateOut = graph.AddLayer<OutputLayer>(2, "cellStateOut");
    Layer* const output = graph.AddLayer<OutputLayer>(3, "output");

    // connect up
    armnn::TensorInfo lstmTensorInfo1({ batchSize, inputSize }, DataType::Float32);
    armnn::TensorInfo lstmTensorInfo2({ batchSize, numUnits}, DataType::Float32);
    armnn::TensorInfo lstmTensorInfo3({ batchSize, outputSize }, DataType::Float32);
    armnn::TensorInfo lstmTensorInfoScratchBuff({ batchSize, numUnits * (layerDesc.m_CifgEnabled ? 3 : 4) },
                                                DataType::Float32);

    Connect(input, layer, lstmTensorInfo1, 0, 0);
    Connect(cellStateIn, layer, lstmTensorInfo2, 0, 1);
    Connect(outputStateIn, layer, lstmTensorInfo3, 0, 2);
    Connect(layer, scratchBuffer, lstmTensorInfoScratchBuff, 0, 0);
    Connect(layer, outputStateOut, lstmTensorInfo3, 1, 0);
    Connect(layer, cellStateOut, lstmTensorInfo2, 2, 0);
    Connect(layer, output, lstmTensorInfo3, 3, 0);
}

}

BOOST_AUTO_TEST_SUITE(Optimizer)
using namespace armnn::optimizations;

BOOST_AUTO_TEST_CASE(OptimizeInversePermutesTest)
{
    armnn::Graph graph;

    auto output = graph.AddLayer<armnn::OutputLayer>(0, "output");

    graph.InsertNewLayer<armnn::InputLayer>(output->GetInputSlot(0), 0, "input");

    // Inserts two permutes, one the inverse of the other.
    graph.InsertNewLayer<armnn::PermuteLayer>(output->GetInputSlot(0),
                                              armnn::PermuteDescriptor({0, 2, 3, 1}),
                                              "perm0231");
    graph.InsertNewLayer<armnn::PermuteLayer>(output->GetInputSlot(0),
                                              armnn::PermuteDescriptor({0, 3, 1, 2}),
                                              "perm0312");

    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::PermuteLayer>,
                             &IsLayerOfType<armnn::PermuteLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(OptimizeInversePermutes()));

    // The permutes are removed.
    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));
}

BOOST_AUTO_TEST_CASE(LSTMValidateTensorShapesFromInputsCIFGDisabledTest)
{
    Graph graph;

    //Helper function creates graph containing LSTM layer with required input and output layers
    CreateLSTMLayerHelper(graph, false);

    //This function used to call ValidateShapesFromInputs();
    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(LSTMValidateTensorShapesFromInputsCIFGEnabledTest)
{
    Graph graph;

    //Helper function creates graph containing LSTM layer with required input and output layers
    CreateLSTMLayerHelper(graph, true);

    //This function used to call ValidateShapesFromInputs();
    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(MovePermuteUpTest)
{
    const armnn::TensorInfo info({ 1, 5, 2, 3 }, armnn::DataType::Float32);
    const armnn::TensorInfo permuted({ 1, 3, 5, 2 }, armnn::DataType::Float32);

    armnn::Graph graph;

    armnn::LayerBindingId inputId = 0;

    armnn::Layer* head = graph.AddLayer<armnn::OutputLayer>(0, "output");

    std::string permuteLayerName = "original_permute";

    // Insert permute
    head = graph.InsertNewLayer<armnn::PermuteLayer>(head->GetInputSlot(0),
                                                     armnn::PermuteDescriptor({ 0, 2, 3, 1 }),
                                                     permuteLayerName.c_str());

    head->GetOutputHandler().SetTensorInfo(permuted);

    // Inserts layers that don't care about data format.
    head = graph.InsertNewLayer<armnn::ActivationLayer>(head->GetInputSlot(0),
                                                        armnn::ActivationDescriptor{}, "");
    head->GetOutputHandler().SetTensorInfo(info);

    head = graph.InsertNewLayer<armnn::AdditionLayer>(head->GetInputSlot(0), "");
    head->GetOutputHandler().SetTensorInfo(info);

    // Inserts input for 2nd input of Addition.
    graph.InsertNewLayer<armnn::InputLayer>(head->GetInputSlot(1), inputId++, "")
        ->GetOutputHandler().SetTensorInfo(info);

    head = graph.InsertNewLayer<armnn::FakeQuantizationLayer>(head->GetInputSlot(0),
                                                              armnn::FakeQuantizationDescriptor{}, "");
    head->GetOutputHandler().SetTensorInfo(info);

    head = graph.InsertNewLayer<armnn::FloorLayer>(head->GetInputSlot(0), "");
    head->GetOutputHandler().SetTensorInfo(info);

    head = graph.InsertNewLayer<armnn::MemCopyLayer>(head->GetInputSlot(0), "");
    head->GetOutputHandler().SetTensorInfo(info);

    head = graph.InsertNewLayer<armnn::MultiplicationLayer>(head->GetInputSlot(0), "");
    head->GetOutputHandler().SetTensorInfo(info);

    // Inserts input for 2nd input of Multiplication.
    graph.InsertNewLayer<armnn::InputLayer>(head->GetInputSlot(1), inputId++, "")
        ->GetOutputHandler().SetTensorInfo(info);

    // Inserts input.
    graph.InsertNewLayer<armnn::InputLayer>(head->GetInputSlot(0), inputId++, "")
        ->GetOutputHandler().SetTensorInfo(info);

    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::MultiplicationLayer>,
                             &IsLayerOfType<armnn::MemCopyLayer>,
                             &IsLayerOfType<armnn::FloorLayer>,
                             &IsLayerOfType<armnn::FakeQuantizationLayer>,
                             &IsLayerOfType<armnn::AdditionLayer>,
                             &IsLayerOfType<armnn::ActivationLayer>,
                             &IsLayerOfType<armnn::PermuteLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(MovePermuteUp()));

    // The permute is moved to the top. New permutes for layers with multiple inputs.
    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::PermuteLayer>,
                             &IsLayerOfType<armnn::PermuteLayer>,
                             &IsLayerOfType<armnn::PermuteLayer>,
                             &IsLayerOfType<armnn::MultiplicationLayer>,
                             &IsLayerOfType<armnn::MemCopyLayer>,
                             &IsLayerOfType<armnn::FloorLayer>,
                             &IsLayerOfType<armnn::FakeQuantizationLayer>,
                             &IsLayerOfType<armnn::AdditionLayer>,
                             &IsLayerOfType<armnn::ActivationLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));

    std::list<std::string> testRelatedLayers = { permuteLayerName };

    BOOST_TEST(CheckRelatedLayers<armnn::PermuteLayer>(graph, testRelatedLayers));
}

BOOST_AUTO_TEST_CASE(PermuteAsReshapeTest)
{
    armnn::Graph graph;

    std::string permuteLayerName = "permute";

    const armnn::TensorInfo infoIn({ 1, 2, 3, 1 }, armnn::DataType::Float32);
    const armnn::TensorInfo infoOut({ 1, 1, 2, 3 }, armnn::DataType::Float32);

    auto output = graph.AddLayer<armnn::OutputLayer>(0, "output");

    graph.InsertNewLayer<armnn::InputLayer>(output->GetInputSlot(0), 0, "input")
        ->GetOutputHandler().SetTensorInfo(infoIn);

    // Inserts permute.
    graph.InsertNewLayer<armnn::PermuteLayer>(output->GetInputSlot(0),
                                              armnn::PermuteDescriptor({ 0, 2, 3, 1 }), permuteLayerName.c_str())
        ->GetOutputHandler().SetTensorInfo(infoOut);

    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::PermuteLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(PermuteAsReshape()));

    // The permute is replaced by an equivalent reshape.

    auto checkReshape = [&infoOut](const armnn::Layer* const layer) -> bool
        {
            const auto reshapeLayer = static_cast<const armnn::ReshapeLayer*>(layer);
            return IsLayerOfType<armnn::ReshapeLayer>(layer) &&
                   (reshapeLayer->GetParameters().m_TargetShape == infoOut.GetShape()) &&
                   (reshapeLayer->GetOutputHandler().GetTensorInfo().GetShape() == infoOut.GetShape());
        };

    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             checkReshape,
                             &IsLayerOfType<armnn::OutputLayer>));


    std::list<std::string> testRelatedLayers = { permuteLayerName };
    BOOST_TEST(CheckRelatedLayers<armnn::ReshapeLayer>(graph, testRelatedLayers));
}

BOOST_AUTO_TEST_CASE(OptimizeConsecutiveReshapesTest)
{
    armnn::Graph graph;

    const armnn::TensorInfo info0({ 1, 2, 3, 5 }, armnn::DataType::Float32);

    auto output = graph.AddLayer<armnn::OutputLayer>(0, "output");
    auto input = graph.InsertNewLayer<armnn::InputLayer>(output->GetInputSlot(0), 0, "input");

    input->GetOutputHandler().SetTensorInfo(info0);

    {
        // Inserts two reshapes.
        const armnn::TensorInfo info1({1, 30, 1, 1}, armnn::DataType::Float32);
        const armnn::TensorInfo info2({1, 2, 1, 15}, armnn::DataType::Float32);

        std::string reshape1Name = "reshape1";
        std::string reshape2Name = "reshape2";

        auto reshape1 = graph.InsertNewLayer<armnn::ReshapeLayer>(output->GetInputSlot(0),
                                                                  armnn::ReshapeDescriptor{ info1.GetShape() },
                                                                  reshape1Name.c_str());
        auto reshape2 = graph.InsertNewLayer<armnn::ReshapeLayer>(output->GetInputSlot(0),
                                                                  armnn::ReshapeDescriptor{ info2.GetShape() },
                                                                  reshape2Name.c_str());

        reshape1->GetOutputHandler().SetTensorInfo(info1);
        reshape2->GetOutputHandler().SetTensorInfo(info2);

        BOOST_TEST(CheckSequence(graph.cbegin(),
                                 graph.cend(),
                                 &IsLayerOfType<armnn::InputLayer>,
                                 &IsLayerOfType<armnn::ReshapeLayer>,
                                 &IsLayerOfType<armnn::ReshapeLayer>,
                                 &IsLayerOfType<armnn::OutputLayer>));

        armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(OptimizeConsecutiveReshapes()));

        auto checkReshape = [&info2](const armnn::Layer* const layer) -> bool
            {
                const auto reshapeLayer = static_cast<const armnn::ReshapeLayer*>(layer);
                return IsLayerOfType<armnn::ReshapeLayer>(layer) &&
                    (reshapeLayer->GetParameters().m_TargetShape == info2.GetShape()) &&
                    (reshapeLayer->GetOutputHandler().GetTensorInfo().GetShape() == info2.GetShape());
            };

        // The two reshapes are replaced by a single equivalent reshape.
        BOOST_TEST(CheckSequence(graph.cbegin(),
                                 graph.cend(),
                                 &IsLayerOfType<armnn::InputLayer>,
                                 checkReshape,
                                 &IsLayerOfType<armnn::OutputLayer>));

        // Check the new reshape layer has the other two reshapes as related layers
        std::list<std::string> testRelatedLayers = { reshape2Name, reshape1Name };

        BOOST_TEST(CheckRelatedLayers<armnn::ReshapeLayer>(graph, testRelatedLayers));
    }

    {
        // Inserts a reshape to the input shape.
        auto reshapeToIn = graph.InsertNewLayer<armnn::ReshapeLayer>(output->GetInputSlot(0),
                                                                     armnn::ReshapeDescriptor{ info0.GetShape() },
                                                                     "reshapeToIn");

        reshapeToIn->GetOutputHandler().SetTensorInfo(info0);

        armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(OptimizeConsecutiveReshapes()));

        // The two reshapes are removed.
        BOOST_TEST(CheckSequence(graph.cbegin(),
                                 graph.cend(),
                                 &IsLayerOfType<armnn::InputLayer>,
                                 &IsLayerOfType<armnn::OutputLayer>));
    }
}

BOOST_AUTO_TEST_CASE(SquashEqualSiblingsTest)
{
    armnn::Graph graph;

    armnn::LayerBindingId outputId = 0;

    const armnn::TensorInfo info({ 1, 2, 3, 5 }, armnn::DataType::Float32);
    const armnn::TensorInfo permuted({ 1, 5, 2, 3 }, armnn::DataType::Float32);

    auto input = graph.AddLayer<armnn::InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(info);

    // Inserts equal permutes, equal reshapes and something else.
    const armnn::PermuteDescriptor permDesc({ 0, 2, 3, 1 });
    const armnn::ReshapeDescriptor reshapeDesc{ { 1, 3, 1, 5 } };

    armnn::Layer* layer;

    layer = graph.AddLayer<armnn::PermuteLayer>(permDesc, "");
    layer->GetOutputSlot().SetTensorInfo(permuted);
    layer->GetOutputSlot().Connect(graph.AddLayer<armnn::OutputLayer>(outputId++, "")->GetInputSlot(0));
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));

    layer = graph.AddLayer<armnn::ReshapeLayer>(reshapeDesc, "");
    layer->GetOutputSlot().Connect(graph.AddLayer<armnn::OutputLayer>(outputId++, "")->GetInputSlot(0));
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));

    layer = graph.AddLayer<armnn::FloorLayer>("");
    layer->GetOutputSlot().Connect(graph.AddLayer<armnn::OutputLayer>(outputId++, "")->GetInputSlot(0));
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));

    layer = graph.AddLayer<armnn::ReshapeLayer>(reshapeDesc, "");
    layer->GetOutputSlot().Connect(graph.AddLayer<armnn::OutputLayer>(outputId++, "")->GetInputSlot(0));
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));

    layer = graph.AddLayer<armnn::PermuteLayer>(permDesc, "");
    layer->GetOutputSlot().SetTensorInfo(permuted);
    layer->GetOutputSlot().Connect(graph.AddLayer<armnn::OutputLayer>(outputId++, "")->GetInputSlot(0));
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));

    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::PermuteLayer>,
                             &IsLayerOfType<armnn::ReshapeLayer>,
                             &IsLayerOfType<armnn::FloorLayer>,
                             &IsLayerOfType<armnn::ReshapeLayer>,
                             &IsLayerOfType<armnn::PermuteLayer>,
                             &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(SquashEqualPermuteSiblings(),
                                                            SquashEqualReshapeSiblings()));

    // The permutes and reshapes are squashed.

    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::PermuteLayer>,
                             &IsLayerOfType<armnn::ReshapeLayer>,
                             &IsLayerOfType<armnn::FloorLayer>,
                             &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));
}

BOOST_AUTO_TEST_CASE(ConvertConstantsHalfToFloatTest)
{
    armnn::Graph graph;

    const armnn::TensorInfo info({ 1,1,1,2 }, armnn::DataType::Float32);

    // Create the half precision input data
    unsigned int dims[] = { 4,1,1,1 };
    std::vector<float> convWeightsData{1.f, 2.f, 3.f, 4.f};
    std::vector<uint16_t> halfWeights(4);
    armnnUtils::FloatingPointConverter::ConvertFloat32To16(convWeightsData.data(),
                                                           convWeightsData.size(),
                                                           halfWeights.data());
    armnn::ConstTensor weights(armnn::TensorInfo(4, dims, armnn::DataType::Float16), halfWeights);

    //Create the simple test network
    auto input = graph.AddLayer<armnn::InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(info);

    auto fc = graph.AddLayer<armnn::FullyConnectedLayer>(armnn::FullyConnectedDescriptor(), "fc");
    fc->m_Weight = std::make_unique<armnn::ScopedCpuTensorHandle>(weights);
    fc->GetOutputSlot().SetTensorInfo(info);

    auto output = graph.AddLayer<armnn::OutputLayer>(1, "output");

    //Connect up the layers
    input->GetOutputSlot().Connect(fc->GetInputSlot(0));
    fc->GetOutputSlot().Connect(output->GetInputSlot(0));

    //Test the tensor info is correct.
    BOOST_CHECK(fc->m_Weight->GetTensorInfo().GetDataType() == armnn::DataType::Float16);

    // Run the optimizer
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(ConvertConstantsHalfToFloat()));

    //Test the tensor info is correct.
    BOOST_CHECK(fc->m_Weight->GetTensorInfo().GetDataType() == armnn::DataType::Float32);

    // Now test the data matches float32 data
    float* data = fc->m_Weight->GetTensor<float>();
    BOOST_CHECK(1.0f == data[0]);
    BOOST_CHECK(2.0f == data[1]);
    BOOST_CHECK(3.0f == data[2]);
    BOOST_CHECK(4.0f == data[3]);
}

BOOST_AUTO_TEST_CASE(ConvertConstantsFloatToHalfTest)
{
    armnn::Graph graph;

    const armnn::TensorInfo info({ 1, 1, 1, 2 }, armnn::DataType::Float16);

    // Create const tensor from fp32 data
    unsigned int dims[] = { 4, 1, 1, 1 };
    std::vector<float> floatWeights{ 1.0f, 2.0f, 3.0f, 4.0f };
    armnn::ConstTensor weights(armnn::TensorInfo(4, dims, armnn::DataType::Float32), floatWeights);

    // Create simple test network
    auto input = graph.AddLayer<armnn::InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(info);

    auto fc = graph.AddLayer<armnn::FullyConnectedLayer>(armnn::FullyConnectedDescriptor(), "fc");
    fc->m_Weight = std::make_unique<armnn::ScopedCpuTensorHandle>(weights);
    fc->GetOutputSlot().SetTensorInfo(info);

    auto output = graph.AddLayer<armnn::OutputLayer>(1, "output");

    // Connect up the layers
    input->GetOutputSlot().Connect(fc->GetInputSlot(0));
    fc->GetOutputSlot().Connect(output->GetInputSlot(0));

    // Check tensor data type before conversion
    BOOST_CHECK(fc->m_Weight->GetTensorInfo().GetDataType() == armnn::DataType::Float32);

    // Run the optimizer
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(ConvertConstantsFloatToHalf()));

    // Check tensor data type after conversion
    BOOST_CHECK(fc->m_Weight->GetTensorInfo().GetDataType() == armnn::DataType::Float16);

    // Check whether data matches expected fp16 data
    Half* data = fc->m_Weight->GetTensor<Half>();
    BOOST_CHECK(data[0] == Half(1.0f));
    BOOST_CHECK(data[1] == Half(2.0f));
    BOOST_CHECK(data[2] == Half(3.0f));
    BOOST_CHECK(data[3] == Half(4.0f));
}

BOOST_AUTO_TEST_CASE(OptimizeInverseConversionsTest)
{
    armnn::Graph graph;

    auto output = graph.AddLayer<armnn::OutputLayer>(0, "output");

    graph.InsertNewLayer<armnn::InputLayer>(output->GetInputSlot(0), 0, "input");

    // Fp32ToFp16 conversion followed by an inverse Fp16ToFp32 conversion
    graph.InsertNewLayer<armnn::ConvertFp32ToFp16Layer>(output->GetInputSlot(0), "convert1");
    graph.InsertNewLayer<armnn::ConvertFp16ToFp32Layer>(output->GetInputSlot(0), "convert2");

    graph.InsertNewLayer<armnn::Convolution2dLayer>(output->GetInputSlot(0), Convolution2dDescriptor(), "conv");

    // Fp16ToFp32 conversion followed by an inverse Fp32ToFp16 conversion
    graph.InsertNewLayer<armnn::ConvertFp16ToFp32Layer>(output->GetInputSlot(0), "convert3");
    graph.InsertNewLayer<armnn::ConvertFp32ToFp16Layer>(output->GetInputSlot(0), "convert4");

    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::ConvertFp32ToFp16Layer>,
                             &IsLayerOfType<armnn::ConvertFp16ToFp32Layer>,
                             &IsLayerOfType<armnn::Convolution2dLayer>,
                             &IsLayerOfType<armnn::ConvertFp16ToFp32Layer>,
                             &IsLayerOfType<armnn::ConvertFp32ToFp16Layer>,
                             &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(OptimizeInverseConversionsFp16(),
                                                           OptimizeInverseConversionsFp32()));

    // Check that all consecutive inverse conversions are removed
    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::Convolution2dLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));
}

BOOST_AUTO_TEST_CASE(InsertConvertersTest)
{
    const armnn::TensorInfo info({ 1, 5, 2, 3 }, armnn::DataType::Float16);

    armnn::Graph graph;

    armnn::LayerBindingId inputId = 0;

    armnn::Layer* head = graph.AddLayer<armnn::OutputLayer>(0, "output");

    head = graph.InsertNewLayer<armnn::AdditionLayer>(head->GetInputSlot(0), "");
    head->GetOutputHandler().SetTensorInfo(info);

    graph.InsertNewLayer<armnn::InputLayer>(head->GetInputSlot(1), inputId++, "")
        ->GetOutputHandler().SetTensorInfo(info);

    head = graph.InsertNewLayer<armnn::FloorLayer>(head->GetInputSlot(0), "");
    head->GetOutputHandler().SetTensorInfo(info);

    head = graph.InsertNewLayer<armnn::MemCopyLayer>(head->GetInputSlot(0), "");
    head->GetOutputHandler().SetTensorInfo(info);

    graph.InsertNewLayer<armnn::InputLayer>(head->GetInputSlot(0), inputId++, "")
        ->GetOutputHandler().SetTensorInfo(info);

    // Check graph layer sequence before inserting convert layers
    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::MemCopyLayer>,
                             &IsLayerOfType<armnn::FloorLayer>,
                             &IsLayerOfType<armnn::AdditionLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));

    // Check layers have Float16 DataType
    for (auto& layer : graph)
    {
        if(layer->GetType()==LayerType::Floor || layer->GetType() == LayerType::Addition)
        {
            BOOST_ASSERT(layer->GetOutputSlot(0).GetTensorInfo().GetDataType() == DataType::Float16);
            BOOST_ASSERT(layer->GetDataType() == DataType::Float16);
        }
    }

    // Insert convert layers either side of unsupported layer
    for (auto& layer : graph)
    {
        if(layer->GetType()==LayerType::Floor || layer->GetType() == LayerType::Addition)
        {
            InsertConvertFp16ToFp32LayersBefore(graph, *layer);
            InsertConvertFp32ToFp16LayersAfter(graph, *layer);
        }
    }

    // Check layers have correct DataType after inserting convert layers
    for (auto& layer : graph)
    {
        if (layer->GetType()==LayerType::Floor || layer->GetType() == LayerType::Addition)
        {
            BOOST_ASSERT(layer->GetOutputSlot(0).GetTensorInfo().GetDataType() == DataType::Float32);
            BOOST_ASSERT(layer->GetDataType() == DataType::Float32);
        }
        else if (layer->GetType() == LayerType::ConvertFp16ToFp32)
        {
            BOOST_ASSERT(layer->GetOutputSlot(0).GetTensorInfo().GetDataType() == DataType::Float32);
            BOOST_ASSERT(layer->GetDataType() == DataType::Float16);
        }
        else if (layer->GetType() == LayerType::ConvertFp32ToFp16)
        {
            BOOST_ASSERT(layer->GetOutputSlot(0).GetTensorInfo().GetDataType() == DataType::Float16);
            BOOST_ASSERT(layer->GetDataType() == DataType::Float32);
        }
    }

    // Check sequence of layers after inserting convert layers
    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::ConvertFp16ToFp32Layer>,
                             &IsLayerOfType<armnn::MemCopyLayer>,
                             &IsLayerOfType<armnn::ConvertFp16ToFp32Layer>,
                             &IsLayerOfType<armnn::FloorLayer>,
                             &IsLayerOfType<armnn::ConvertFp32ToFp16Layer>,
                             &IsLayerOfType<armnn::ConvertFp16ToFp32Layer>,
                             &IsLayerOfType<armnn::AdditionLayer>,
                             &IsLayerOfType<armnn::ConvertFp32ToFp16Layer>,
                             &IsLayerOfType<armnn::OutputLayer>));
}

BOOST_AUTO_TEST_CASE(Fp32NetworkToFp16OptimizationTest)
{
    armnn::Graph graph;

    const armnn::TensorInfo infoFP32({ 2,2,1,3 }, armnn::DataType::Float32);

    // Create the simple test network
    auto input = graph.AddLayer<armnn::InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(infoFP32);

    auto floor = graph.AddLayer<armnn::FloorLayer>("floor");
    floor->GetOutputSlot().SetTensorInfo(infoFP32);

    auto output = graph.AddLayer<armnn::OutputLayer>(1, "output");

    // Connect up the layers
    input->GetOutputSlot().Connect(floor->GetInputSlot(0));
    floor->GetOutputSlot().Connect(output->GetInputSlot(0));

    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::FloorLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));

    // Run the optimizer
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(Fp32NetworkToFp16Converter()));

    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::ConvertFp32ToFp16Layer>,
                             &IsLayerOfType<armnn::FloorLayer>,
                             &IsLayerOfType<armnn::ConvertFp16ToFp32Layer>,
                             &IsLayerOfType<armnn::OutputLayer>));
}

BOOST_AUTO_TEST_CASE(InsertDebugOptimizationTest)
{
    armnn::Graph graph;

    const armnn::TensorInfo info({ 2,2,1,3 }, armnn::DataType::Float32);

    // Create the simple test network
    auto input = graph.AddLayer<armnn::InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(info);

    auto floor = graph.AddLayer<armnn::FloorLayer>("floor");
    floor->GetOutputSlot().SetTensorInfo(info);

    auto output = graph.AddLayer<armnn::OutputLayer>(1, "output");

    // Connect up the layers
    input->GetOutputSlot().Connect(floor->GetInputSlot(0));
    floor->GetOutputSlot().Connect(output->GetInputSlot(0));

    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::FloorLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));

    // Run the optimizer
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(InsertDebugLayer()));

    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::DebugLayer>,
                             &IsLayerOfType<armnn::FloorLayer>,
                             &IsLayerOfType<armnn::DebugLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));
}

void CreateConvolution2dGraph(Graph &graph, const unsigned int* inputShape,
                              const unsigned int* weightsShape, const unsigned int* outputShape,
                              DataLayout dataLayout = DataLayout::NCHW)
{
    armnn::TensorInfo inputInfo(4, inputShape, DataType::Float32);
    armnn::TensorInfo outputInfo(4, outputShape, DataType::Float32);

    std::vector<float> weightsVector(90);
    armnn::ConstTensor weights(armnn::TensorInfo(4, weightsShape, armnn::DataType::Float32), weightsVector);

    Convolution2dDescriptor desc;
    desc.m_BiasEnabled = false;
    desc.m_StrideX = 1;
    desc.m_StrideY = 1;
    desc.m_DataLayout = dataLayout;

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    Convolution2dLayer* layer = graph.AddLayer<Convolution2dLayer>(desc, "conv2d");
    layer->m_Weight = std::make_unique<armnn::ScopedCpuTensorHandle>(weights);
    layer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot().Connect(output->GetInputSlot(0));
}

BOOST_AUTO_TEST_CASE(Conv2dValidateTensorShapesFromInputs)
{
    Graph graph;
    const unsigned int inputShape[] = { 1, 3, 8, 16 };
    const unsigned int weightsShape[] = { 2, 3, 5, 3 };
    const unsigned int outputShape[] = { 1, 2, 4, 14 };
    CreateConvolution2dGraph(graph, inputShape, weightsShape, outputShape);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(Conv2dValidateTensorShapesFromInputsNhwc)
{
    Graph graph;
    const unsigned int inputShape[] = { 1, 8, 16, 3 };
    const unsigned int weightsShape[] = { 2, 5, 3, 3 };
    const unsigned int outputShape[] = { 1, 4, 14, 2 };
    CreateConvolution2dGraph(graph, inputShape, weightsShape, outputShape, DataLayout::NHWC);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

void CreateDepthwiseConvolution2dGraph(Graph &graph, const unsigned int* inputShape,
                              const unsigned int* weightsShape, const unsigned int* outputShape,
                              DataLayout dataLayout = DataLayout::NCHW)
{
    armnn::TensorInfo inputInfo(4, inputShape, DataType::Float32);
    armnn::TensorInfo outputInfo(4, outputShape, DataType::Float32);

    std::vector<float> weightsVector(18);
    armnn::ConstTensor weights(armnn::TensorInfo(4, weightsShape, armnn::DataType::Float32), weightsVector);

    DepthwiseConvolution2dDescriptor desc;
    desc.m_BiasEnabled = false;
    desc.m_StrideX = 1;
    desc.m_StrideY = 1;
    desc.m_DataLayout = dataLayout;

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    DepthwiseConvolution2dLayer* layer = graph.AddLayer<DepthwiseConvolution2dLayer>(desc, "depthwiseConv2d");
    layer->m_Weight = std::make_unique<armnn::ScopedCpuTensorHandle>(weights);
    layer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot().Connect(output->GetInputSlot(0));
}

BOOST_AUTO_TEST_CASE(DepthwiseConv2dValidateTensorShapesFromInputs)
{
    Graph graph;
    const unsigned int inputShape[] = { 1, 2, 3, 3 };
    const unsigned int weightsShape[] = { 1, 2, 3, 3 };
    const unsigned int outputShape[] = { 1, 2, 1, 1 };
    CreateDepthwiseConvolution2dGraph(graph, inputShape, weightsShape, outputShape);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(DepthwiseConv2dValidateTensorShapesFromInputsNhwc)
{
    Graph graph;
    const unsigned int inputShape[] = { 1, 3, 3, 2 };
    const unsigned int weightsShape[] = { 1, 2, 3, 3 };
    const unsigned int outputShape[] = { 1, 1, 1, 2 };
    CreateDepthwiseConvolution2dGraph(graph, inputShape, weightsShape, outputShape, DataLayout::NHWC);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

void CreatePooling2dGraph(Graph &graph, const unsigned int* inputShape,  const unsigned int* outputShape,
                          DataLayout dataLayout = DataLayout::NCHW)
{
    armnn::TensorInfo inputInfo(4, inputShape, DataType::Float32);
    armnn::TensorInfo outputInfo(4, outputShape, DataType::Float32);

    Pooling2dDescriptor desc;
    desc.m_PoolType = armnn::PoolingAlgorithm::Average;
    desc.m_PoolWidth = desc.m_PoolHeight = 100;
    desc.m_StrideX = desc.m_StrideY = 5;
    desc.m_PadLeft = 50;
    desc.m_PadRight = 50;
    desc.m_PadTop = 50;
    desc.m_PadBottom = 50;
    desc.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    desc.m_DataLayout = dataLayout;

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    Pooling2dLayer* layer = graph.AddLayer<Pooling2dLayer>(desc, "pooling2d");
    layer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot().Connect(output->GetInputSlot(0));
}

BOOST_AUTO_TEST_CASE(Pooling2dValidateTensorShapesFromInputs)
{
    Graph graph;
    const unsigned int inputShape[] = { 5, 3, 52, 60 };
    const unsigned int outputShape[] = { 5, 3, 11, 13 };
    CreatePooling2dGraph(graph, inputShape, outputShape, DataLayout::NCHW);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(Pooling2dValidateTensorShapesFromInputsNhwc)
{
    Graph graph;
    const unsigned int inputShape[] = { 5, 52, 60, 3 };
    const unsigned int outputShape[] = { 5, 11, 13, 3 };
    CreatePooling2dGraph(graph, inputShape, outputShape, DataLayout::NHWC);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

void CreateResizeBilinearGraph(Graph &graph, const unsigned int* inputShape,  const unsigned int* outputShape,
                               DataLayout dataLayout = DataLayout::NCHW)
{
    TensorInfo inputInfo(4, inputShape, DataType::Float32);
    TensorInfo outputInfo(4, outputShape, DataType::Float32);

    ResizeDescriptor desc;
    desc.m_Method       = ResizeMethod::Bilinear;
    desc.m_TargetHeight = 3;
    desc.m_TargetWidth  = 4;
    desc.m_DataLayout   = dataLayout;

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    ResizeLayer* layer = graph.AddLayer<ResizeLayer>(desc, "resizeBilinear");
    layer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot().Connect(output->GetInputSlot(0));
}

BOOST_AUTO_TEST_CASE(ResizeBilinearValidateTensorShapesFromInputs)
{
    Graph graph;
    const unsigned int inputShape[] = { 1, 2, 4, 5 };
    const unsigned int outputShape[] = { 1, 2, 3, 4 };
    CreateResizeBilinearGraph(graph, inputShape, outputShape);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(ResizeBilinearValidateTensorShapesFromInputsNhwc)
{
    Graph graph;
    const unsigned int inputShape[] = { 1, 4, 5, 2 };
    const unsigned int outputShape[] = { 1, 3, 4, 2 };
    CreateResizeBilinearGraph(graph, inputShape, outputShape, DataLayout::NHWC);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}


void CreateGatherGraph(Graph& graph, const armnn::TensorInfo& paramsInfo, const armnn::TensorInfo& indicesInfo,
                       const armnn::TensorInfo& outputInfo)
{
    Layer* input0 = graph.AddLayer<InputLayer>(0, "params");
    input0->GetOutputSlot().SetTensorInfo(paramsInfo);

    Layer* input1 = graph.AddLayer<InputLayer>(1, "indices");
    input1->GetOutputSlot().SetTensorInfo(indicesInfo);

    GatherLayer* layer = graph.AddLayer<GatherLayer>("gather");
    layer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");
    input0->GetOutputSlot().Connect(layer->GetInputSlot(0));
    input1->GetOutputSlot().Connect(layer->GetInputSlot(1));
    layer->GetOutputSlot().Connect(output->GetInputSlot(0));
}

BOOST_AUTO_TEST_CASE(GatherValidateTensorShapesFromInputs)
{
    Graph graph;
    armnn::TensorInfo paramsInfo({10, 5}, DataType::Float32);
    armnn::TensorInfo indicesInfo({3}, DataType::Signed32);
    armnn::TensorInfo outputInfo({3, 5}, DataType::Float32);

    CreateGatherGraph(graph, paramsInfo, indicesInfo, outputInfo);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(GatherValidateTensorShapesFromInputs1DParams)
{
    Graph graph;
    armnn::TensorInfo paramsInfo({8}, DataType::Float32);
    armnn::TensorInfo indicesInfo({5}, DataType::Signed32);
    armnn::TensorInfo outputInfo( {5}, DataType::Float32);

    CreateGatherGraph(graph, paramsInfo, indicesInfo, outputInfo);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(GatherValidateTensorShapesFromInputsMultiDimIndices)
{
    Graph graph;
    armnn::TensorInfo paramsInfo({3, 2, 5}, DataType::Float32);
    armnn::TensorInfo indicesInfo({2, 2}, DataType::Signed32);
    armnn::TensorInfo outputInfo({2, 2, 2, 5}, DataType::Float32);

    CreateGatherGraph(graph, paramsInfo, indicesInfo, outputInfo);

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(DetectionPostProcessValidateTensorShapes)
{
    Graph graph;
    armnn::TensorInfo boxEncodingsInfo({1, 10, 4}, DataType::QuantisedAsymm8);
    armnn::TensorInfo scoresInfo({1, 10, 4}, DataType::QuantisedAsymm8);
    std::vector<uint8_t> anchorsVector(40);
    armnn::ConstTensor anchors(armnn::TensorInfo({10, 4}, armnn::DataType::QuantisedAsymm8), anchorsVector);

    armnn::TensorInfo detectionBoxesInfo({1, 3, 4}, DataType::QuantisedAsymm8);
    armnn::TensorInfo detectionScoresInfo({1, 3}, DataType::QuantisedAsymm8);
    armnn::TensorInfo detectionClassesInfo({1, 3}, DataType::QuantisedAsymm8);
    armnn::TensorInfo numDetectionInfo({1}, DataType::QuantisedAsymm8);

    Layer* input0 = graph.AddLayer<InputLayer>(0, "boxEncodings");
    input0->GetOutputSlot().SetTensorInfo(boxEncodingsInfo);

    Layer* input1 = graph.AddLayer<InputLayer>(1, "score");
    input1->GetOutputSlot().SetTensorInfo(scoresInfo);

    DetectionPostProcessDescriptor descriptor;
    descriptor.m_MaxDetections = 3;

    DetectionPostProcessLayer* layer = graph.AddLayer<DetectionPostProcessLayer>(descriptor, "detectionPostProcess");
    layer->m_Anchors = std::make_unique<armnn::ScopedCpuTensorHandle>(anchors);
    layer->GetOutputSlot(0).SetTensorInfo(detectionBoxesInfo);
    layer->GetOutputSlot(1).SetTensorInfo(detectionScoresInfo);
    layer->GetOutputSlot(2).SetTensorInfo(detectionClassesInfo);
    layer->GetOutputSlot(3).SetTensorInfo(numDetectionInfo);

    input0->GetOutputSlot().Connect(layer->GetInputSlot(0));
    input1->GetOutputSlot().Connect(layer->GetInputSlot(1));

    BOOST_CHECK_NO_THROW(graph.InferTensorInfos());
}

BOOST_AUTO_TEST_CASE(FoldPadLayerIntoConvolution2dLayer)
{
    Graph graph;
    const unsigned int inputShape[] = { 1, 2, 2, 3 };
    const unsigned int paddedShape[] = { 1, 6, 6, 3 };
    const unsigned int weightsShape[] = { 1, 2, 3, 3 };
    const unsigned int outputShape[] = { 1, 2, 1, 1 };


    armnn::TensorInfo inputInfo(4, inputShape, DataType::Float32);
    armnn::TensorInfo paddedInfo(4, paddedShape, DataType::Float32);
    armnn::TensorInfo outputInfo(4, outputShape, DataType::Float32);

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    PadDescriptor padDescriptor({{ 0, 0 }, { 2, 2 }, { 2, 2 }, { 0, 0 }});

    PadLayer* padLayer = graph.AddLayer<PadLayer>(padDescriptor, "pad");
    padLayer->GetOutputSlot().SetTensorInfo(paddedInfo);

    Convolution2dDescriptor convolution2dDescriptor;
    convolution2dDescriptor.m_BiasEnabled = false;
    convolution2dDescriptor.m_StrideX = 1;
    convolution2dDescriptor.m_StrideY = 1;
    convolution2dDescriptor.m_DataLayout = DataLayout::NHWC;

    std::vector<float> weightsVector(18);
    armnn::ConstTensor weights(armnn::TensorInfo(4, weightsShape, armnn::DataType::Float32), weightsVector);

    Convolution2dLayer* conv2dLayer = graph.AddLayer<Convolution2dLayer>(convolution2dDescriptor,"conv2d");
    conv2dLayer->m_Weight = std::make_unique<armnn::ScopedCpuTensorHandle>(weights);
    conv2dLayer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up layers - input -> pad -> conv2d -> output
    input->GetOutputSlot().Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot().Connect(conv2dLayer->GetInputSlot(0));
    conv2dLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    auto checkSimpleConv2d = [ ](const armnn::Layer* const layer) -> bool
    {
        const auto conv2dLayer = static_cast<const armnn::Convolution2dLayer*>(layer);
        const auto conv2dLayerParams = conv2dLayer->GetParameters();
        return IsLayerOfType<armnn::Convolution2dLayer>(layer) &&
            (layer->GetNameStr() == "conv2d") &&
            (conv2dLayerParams.m_PadLeft == 0) &&
            (conv2dLayerParams.m_PadRight == 0) &&
            (conv2dLayerParams.m_PadTop == 0) &&
            (conv2dLayerParams.m_PadBottom == 0) &&
            (conv2dLayerParams.m_BiasEnabled == false) &&
            (conv2dLayerParams.m_StrideX == 1) &&
            (conv2dLayerParams.m_StrideY == 1) &&
            (conv2dLayerParams.m_DataLayout == DataLayout::NHWC);
    };

    BOOST_TEST(CheckSequence(graph.cbegin(),
        graph.cend(),
        &IsLayerOfType<armnn::InputLayer>,
        &IsLayerOfType<armnn::PadLayer>,
        checkSimpleConv2d,
        &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(FoldPadIntoConvolution2d()));

    auto checkPadFoldedIntoConv2d = [ ](const armnn::Layer* const layer) -> bool
    {
        const auto conv2dLayer = static_cast<const armnn::Convolution2dLayer*>(layer);
        const auto conv2dLayerParams = conv2dLayer->GetParameters();
        return IsLayerOfType<armnn::Convolution2dLayer>(layer) &&
               (layer->GetNameStr() == "folded-pad-into-conv2d") &&
               (conv2dLayerParams.m_PadLeft == 2) &&
               (conv2dLayerParams.m_PadRight == 2) &&
               (conv2dLayerParams.m_PadTop == 2) &&
               (conv2dLayerParams.m_PadBottom == 2) &&
               (conv2dLayerParams.m_BiasEnabled == false) &&
               (conv2dLayerParams.m_StrideX == 1) &&
               (conv2dLayerParams.m_StrideY == 1) &&
               (conv2dLayerParams.m_DataLayout == DataLayout::NHWC);
    };

    BOOST_TEST(CheckSequence(graph.cbegin(),
        graph.cend(),
        &IsLayerOfType<armnn::InputLayer>,
        checkPadFoldedIntoConv2d,
        &IsLayerOfType<armnn::OutputLayer>));
}

BOOST_AUTO_TEST_SUITE_END()
