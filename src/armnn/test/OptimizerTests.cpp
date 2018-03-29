//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include <boost/test/unit_test.hpp>

#include "armnn/ArmNN.hpp"
#include "Graph.hpp"
#include "Optimizer.hpp"

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

/// Check each unary function in Us evaluates true for each correspondent layer in the sequence [first, last)
template <typename U, typename... Us>
bool CheckSequence(const armnn::Graph::ConstIterator first,
                   const armnn::Graph::ConstIterator last,
                   U&& u,
                   Us&&... us)
{
    return u(*first) && CheckSequence(std::next(first), last, us...);
}
}

BOOST_AUTO_TEST_SUITE(Optimizer)

BOOST_AUTO_TEST_CASE(OptimizeInversePermutes)
{
    armnn::Graph graph;

    auto output = graph.AddLayer<armnn::OutputLayer>(0, "output");

    graph.InsertNewLayer<armnn::InputLayer>(output->GetInputSlot(0), 0, "input");

    // Insert two permutes, one the inverse of the other
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

    armnn::Optimizer::Optimize(graph);

    // The permutes are removed
    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));
}

BOOST_AUTO_TEST_CASE(MovePermuteUp)
{
    const armnn::TensorInfo info({ 1, 5, 2, 3 }, armnn::DataType::Float32);
    const armnn::TensorInfo permuted({ 1, 3, 5, 2 }, armnn::DataType::Float32);

    armnn::Graph graph;

    armnn::LayerBindingId inputId = 0;

    armnn::Layer* head = graph.AddLayer<armnn::OutputLayer>(0, "output");

    // Insert permute
    head = graph.InsertNewLayer<armnn::PermuteLayer>(head->GetInputSlot(0),
                                                     armnn::PermuteDescriptor({ 0, 2, 3, 1 }), "");
    head->GetOutputHandler().SetTensorInfo(permuted);

    // Insert layers that don't care about data format
    head = graph.InsertNewLayer<armnn::ActivationLayer>(head->GetInputSlot(0),
                                                        armnn::ActivationDescriptor{}, "");
    head->GetOutputHandler().SetTensorInfo(info);

    head = graph.InsertNewLayer<armnn::AdditionLayer>(head->GetInputSlot(0), "");
    head->GetOutputHandler().SetTensorInfo(info);

    // Insert input for 2nd input of Addition
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

    // Insert input for 2nd input of Multiplication
    graph.InsertNewLayer<armnn::InputLayer>(head->GetInputSlot(1), inputId++, "")
        ->GetOutputHandler().SetTensorInfo(info);

    // Insert input
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

    armnn::Optimizer::Optimize(graph);

    // The permute is moved to the top. New permutes for layers with multiple inputs
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
}

BOOST_AUTO_TEST_CASE(PermuteAsReshape)
{
    armnn::Graph graph;

    const armnn::TensorInfo infoIn({ 1, 2, 3, 1 }, armnn::DataType::Float32);
    const armnn::TensorInfo infoOut({ 1, 1, 2, 3 }, armnn::DataType::Float32);

    auto output = graph.AddLayer<armnn::OutputLayer>(0, "output");

    graph.InsertNewLayer<armnn::InputLayer>(output->GetInputSlot(0), 0, "input")
        ->GetOutputHandler().SetTensorInfo(infoIn);

    // Insert permute
    graph.InsertNewLayer<armnn::PermuteLayer>(output->GetInputSlot(0),
                                              armnn::PermuteDescriptor({ 0, 2, 3, 1 }), "")
        ->GetOutputHandler().SetTensorInfo(infoOut);

    BOOST_TEST(CheckSequence(graph.cbegin(),
                             graph.cend(),
                             &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::PermuteLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Optimize(graph);

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
}

BOOST_AUTO_TEST_CASE(OptimizeConsecutiveReshapes)
{
    armnn::Graph graph;

    const armnn::TensorInfo info0({ 1, 2, 3, 5 }, armnn::DataType::Float32);

    auto output = graph.AddLayer<armnn::OutputLayer>(0, "output");
    auto input = graph.InsertNewLayer<armnn::InputLayer>(output->GetInputSlot(0), 0, "input");

    input->GetOutputHandler().SetTensorInfo(info0);

    {
        // Insert two reshapes
        const armnn::TensorInfo info1({1, 30, 1, 1}, armnn::DataType::Float32);
        const armnn::TensorInfo info2({1, 2, 1, 15}, armnn::DataType::Float32);

        auto reshape1 = graph.InsertNewLayer<armnn::ReshapeLayer>(output->GetInputSlot(0),
                                                                  armnn::ReshapeDescriptor{ info1.GetShape() },
                                                                  "reshape1");
        auto reshape2 = graph.InsertNewLayer<armnn::ReshapeLayer>(output->GetInputSlot(0),
                                                                  armnn::ReshapeDescriptor{ info2.GetShape() },
                                                                  "reshape2");

        reshape1->GetOutputHandler().SetTensorInfo(info1);
        reshape2->GetOutputHandler().SetTensorInfo(info2);

        BOOST_TEST(CheckSequence(graph.cbegin(),
                                 graph.cend(),
                                 &IsLayerOfType<armnn::InputLayer>,
                                 &IsLayerOfType<armnn::ReshapeLayer>,
                                 &IsLayerOfType<armnn::ReshapeLayer>,
                                 &IsLayerOfType<armnn::OutputLayer>));

        armnn::Optimizer::Optimize(graph);

        auto checkReshape = [&info2](const armnn::Layer* const layer) -> bool
            {
                const auto reshapeLayer = static_cast<const armnn::ReshapeLayer*>(layer);
                return IsLayerOfType<armnn::ReshapeLayer>(layer) &&
                    (reshapeLayer->GetParameters().m_TargetShape == info2.GetShape()) &&
                    (reshapeLayer->GetOutputHandler().GetTensorInfo().GetShape() == info2.GetShape());
            };

        // The two reshapes are replaced by a single equivalent reshape
        BOOST_TEST(CheckSequence(graph.cbegin(),
                                 graph.cend(),
                                 &IsLayerOfType<armnn::InputLayer>,
                                 checkReshape,
                                 &IsLayerOfType<armnn::OutputLayer>));
    }

    {
        // Insert a reshape to the input shape
        auto reshapeToIn = graph.InsertNewLayer<armnn::ReshapeLayer>(output->GetInputSlot(0),
                                                                     armnn::ReshapeDescriptor{ info0.GetShape() },
                                                                     "reshapeToIn");

        reshapeToIn->GetOutputHandler().SetTensorInfo(info0);

        armnn::Optimizer::Optimize(graph);

        // The two reshapes are removed
        BOOST_TEST(CheckSequence(graph.cbegin(),
                                 graph.cend(),
                                 &IsLayerOfType<armnn::InputLayer>,
                                 &IsLayerOfType<armnn::OutputLayer>));
    }
}

BOOST_AUTO_TEST_CASE(SquashEqualSiblings)
{
    armnn::Graph graph;

    armnn::LayerBindingId outputId = 0;

    const armnn::TensorInfo info({ 1, 2, 3, 5 }, armnn::DataType::Float32);
    const armnn::TensorInfo permuted({ 1, 5, 2, 3 }, armnn::DataType::Float32);

    auto input = graph.AddLayer<armnn::InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(info);

    // Insert equal permutes, equal reshapes and something else
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

    armnn::Optimizer::Optimize(graph);

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

BOOST_AUTO_TEST_SUITE_END()
