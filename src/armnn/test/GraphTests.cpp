//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "GraphUtils.hpp"

#include <Graph.hpp>
#include <Layer.hpp>

#include <armnn/TypesUtils.hpp>
#include <armnn/Exceptions.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <armnn/backends/IBackendInternal.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/TensorHandleFactoryRegistry.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Graph)

BOOST_AUTO_TEST_CASE(ClassGraph)
{
    armnn::Graph graph;
    BOOST_CHECK_NO_THROW(graph.AddLayer<armnn::InputLayer>(0, "layerA"));
    BOOST_TEST(GraphHasNamedLayer(graph, "layerA"));
}

BOOST_AUTO_TEST_CASE(TopologicalSort)
{
    armnn::Graph graph;

    armnn::ActivationDescriptor activationDefaults;

    BOOST_CHECK_NO_THROW(graph.AddLayer<armnn::InputLayer>(0, "layerA"));
    BOOST_CHECK_NO_THROW(graph.AddLayer<armnn::ActivationLayer>(activationDefaults, "layerB"));
    BOOST_CHECK_NO_THROW(graph.AddLayer<armnn::AdditionLayer>("layerC"));
    BOOST_CHECK_NO_THROW(graph.AddLayer<armnn::OutputLayer>(0, "output"));
    BOOST_CHECK_NO_THROW(graph.AddLayer<armnn::ActivationLayer>(activationDefaults, "layerD"));
    BOOST_CHECK_NO_THROW(graph.AddLayer<armnn::ActivationLayer>(activationDefaults, "layerE"));

    armnn::Layer* const layerA = GetFirstLayerWithName(graph, "layerA");
    armnn::Layer* const layerB = GetFirstLayerWithName(graph, "layerB");
    armnn::Layer* const layerC = GetFirstLayerWithName(graph, "layerC");
    armnn::Layer* const layerO = GetFirstLayerWithName(graph, "output");
    armnn::Layer* const layerE = GetFirstLayerWithName(graph, "layerE");
    armnn::Layer* const layerD = GetFirstLayerWithName(graph, "layerD");

    // Simple graph which branches and rejoins.
    //    A
    //   / \'
    //  D   E
    //   \  |
    //    \ B
    //     \|
    //      C
    layerA->GetOutputSlot(0).Connect(layerD->GetInputSlot(0));
    layerA->GetOutputSlot(0).Connect(layerE->GetInputSlot(0));
    layerE->GetOutputSlot(0).Connect(layerB->GetInputSlot(0));
    layerD->GetOutputSlot(0).Connect(layerC->GetInputSlot(0));
    layerB->GetOutputSlot(0).Connect(layerC->GetInputSlot(1));
    layerC->GetOutputSlot(0).Connect(layerO->GetInputSlot(0));

    // check order is valid
    BOOST_TEST(CheckOrder(graph, layerA, layerD));
    BOOST_TEST(CheckOrder(graph, layerA, layerE));
    BOOST_TEST(CheckOrder(graph, layerD, layerC));
    BOOST_TEST(CheckOrder(graph, layerE, layerB));
    BOOST_TEST(CheckOrder(graph, layerB, layerC));
}

BOOST_AUTO_TEST_CASE(InsertNewLayerBefore)
{
    armnn::Graph graph;
    armnn::TensorInfo tensorInfo({ 1, 1, 1, 1 }, armnn::DataType::Float32);

    std::vector<armnn::Layer*> order;

    armnn::ActivationDescriptor activationDefaults;
    BOOST_CHECK_NO_THROW(graph.AddLayer<armnn::InputLayer>(0, "layerA"));
    BOOST_CHECK_NO_THROW(graph.AddLayer<armnn::ActivationLayer>(activationDefaults, "layerB"));
    BOOST_CHECK_NO_THROW(graph.AddLayer<armnn::ActivationLayer>(activationDefaults, "layerC"));
    BOOST_CHECK_NO_THROW(graph.AddLayer<armnn::AdditionLayer>("layerD"));
    BOOST_CHECK_NO_THROW(graph.AddLayer<armnn::OutputLayer>(0, "output"));

    armnn::Layer* const layerA = GetFirstLayerWithName(graph, "layerA");
    armnn::Layer* const layerB = GetFirstLayerWithName(graph, "layerB");
    armnn::Layer* const layerC = GetFirstLayerWithName(graph, "layerC");
    armnn::Layer* const layerD = GetFirstLayerWithName(graph, "layerD");
    armnn::Layer* const layerO = GetFirstLayerWithName(graph, "output");

    //    A
    //   / \'
    //  B   C
    //   \ /
    //    D
    layerA->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    layerB->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    layerC->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    layerD->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    layerA->GetOutputSlot(0).Connect(layerB->GetInputSlot(0));
    layerA->GetOutputSlot(0).Connect(layerC->GetInputSlot(0));
    layerB->GetOutputSlot(0).Connect(layerD->GetInputSlot(0));
    layerC->GetOutputSlot(0).Connect(layerD->GetInputSlot(1));
    layerD->GetOutputSlot(0).Connect(layerO->GetInputSlot(0));

    // Checks order is valid.
    BOOST_TEST(CheckOrder(graph, layerA, layerB));
    BOOST_TEST(CheckOrder(graph, layerA, layerC));
    BOOST_TEST(CheckOrder(graph, layerB, layerD));
    BOOST_TEST(CheckOrder(graph, layerC, layerD));

    //    A
    //   / \'
    //  B   C
    //   \  |
    //    \ E
    //     \|
    //      D
    BOOST_CHECK_NO_THROW(graph.InsertNewLayer<armnn::ActivationLayer>(layerD->GetInputSlot(1),
                                                                      activationDefaults,
                                                                      "layerE"));

    armnn::Layer* const layerE = GetFirstLayerWithName(graph, "layerE");

    // Checks order is valid.
    BOOST_TEST(CheckOrder(graph, layerA, layerB));
    BOOST_TEST(CheckOrder(graph, layerA, layerC));
    BOOST_TEST(CheckOrder(graph, layerB, layerD));
    BOOST_TEST(CheckOrder(graph, layerC, layerE));
    BOOST_TEST(CheckOrder(graph, layerE, layerD));

    //      A
    //     /|
    //    / F
    //   /  |
    //  B   C
    //   \  |
    //    \ E
    //     \|
    //      D
    BOOST_CHECK_NO_THROW(graph.InsertNewLayer<armnn::ActivationLayer>(layerC->GetInputSlot(0),
                                                                      activationDefaults,
                                                                      "layerF"));

    armnn::Layer* const layerF = GetFirstLayerWithName(graph, "layerF");

    // Checks order is valid.
    BOOST_TEST(CheckOrder(graph, layerA, layerB));
    BOOST_TEST(CheckOrder(graph, layerA, layerF));
    BOOST_TEST(CheckOrder(graph, layerF, layerC));
    BOOST_TEST(CheckOrder(graph, layerB, layerD));
    BOOST_TEST(CheckOrder(graph, layerC, layerE));
    BOOST_TEST(CheckOrder(graph, layerE, layerD));
}

BOOST_AUTO_TEST_CASE(InsertNewLayerAfter)
{
    armnn::Graph graph;
    armnn::TensorInfo tensorInfo({ 1, 1, 1, 1 }, armnn::DataType::Float32);

    std::vector<armnn::Layer*> order;

    armnn::ActivationDescriptor activationDefaults;
    BOOST_CHECK_NO_THROW(graph.AddLayer<armnn::InputLayer>(0, "layerA"));
    BOOST_CHECK_NO_THROW(graph.AddLayer<armnn::ActivationLayer>(activationDefaults, "layerB"));
    BOOST_CHECK_NO_THROW(graph.AddLayer<armnn::ActivationLayer>(activationDefaults, "layerC"));
    BOOST_CHECK_NO_THROW(graph.AddLayer<armnn::AdditionLayer>("layerD"));
    BOOST_CHECK_NO_THROW(graph.AddLayer<armnn::OutputLayer>(0, "output"));

    armnn::Layer* const layerA = GetFirstLayerWithName(graph, "layerA");
    armnn::Layer* const layerB = GetFirstLayerWithName(graph, "layerB");
    armnn::Layer* const layerC = GetFirstLayerWithName(graph, "layerC");
    armnn::Layer* const layerD = GetFirstLayerWithName(graph, "layerD");
    armnn::Layer* const layerO = GetFirstLayerWithName(graph, "output");

    //    A
    //   / \'
    //  B   C
    //   \ /
    //    D
    layerA->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    layerB->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    layerC->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    layerD->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    layerA->GetOutputSlot(0).Connect(layerB->GetInputSlot(0));
    layerA->GetOutputSlot(0).Connect(layerC->GetInputSlot(0));
    layerB->GetOutputSlot(0).Connect(layerD->GetInputSlot(0));
    layerC->GetOutputSlot(0).Connect(layerD->GetInputSlot(1));
    layerD->GetOutputSlot(0).Connect(layerO->GetInputSlot(0));

    // Checks order is valid.
    BOOST_TEST(CheckOrder(graph, layerA, layerB));
    BOOST_TEST(CheckOrder(graph, layerA, layerC));
    BOOST_TEST(CheckOrder(graph, layerB, layerD));
    BOOST_TEST(CheckOrder(graph, layerC, layerD));

    //    A
    //   / \'
    //  B   C
    //   \  |
    //    \ E
    //     \|
    //      D
    BOOST_CHECK_NO_THROW(graph.InsertNewLayer<armnn::ActivationLayer>(layerC->GetOutputSlot(),
                                                                      activationDefaults,
                                                                      "layerE"));

    armnn::Layer* const layerE = GetFirstLayerWithName(graph, "layerE");

    // Checks order is valid.
    BOOST_TEST(CheckOrder(graph, layerA, layerB));
    BOOST_TEST(CheckOrder(graph, layerA, layerC));
    BOOST_TEST(CheckOrder(graph, layerB, layerD));
    BOOST_TEST(CheckOrder(graph, layerC, layerE));
    BOOST_TEST(CheckOrder(graph, layerE, layerD));


    //    A
    //    |
    //    F
    //   / \'
    //  B   C
    //  \   |
    //   \  E
    //    \ /
    //     D
    BOOST_CHECK_NO_THROW(graph.InsertNewLayer<armnn::ActivationLayer>(layerA->GetOutputSlot(),
                                                                      activationDefaults,
                                                                      "layerF"));

    armnn::Layer* const layerF = GetFirstLayerWithName(graph, "layerF");

    // Checks order is valid.
    BOOST_TEST(CheckOrder(graph, layerA, layerF));
    BOOST_TEST(CheckOrder(graph, layerF, layerB));
    BOOST_TEST(CheckOrder(graph, layerF, layerC));
    BOOST_TEST(CheckOrder(graph, layerB, layerD));
    BOOST_TEST(CheckOrder(graph, layerC, layerE));
    BOOST_TEST(CheckOrder(graph, layerE, layerD));
}

namespace
{
    using Edge = std::pair<const armnn::Layer*, const armnn::Layer*>;
}

static std::vector<Edge> GetEdgeList(const armnn::Graph& graph)
{
    std::vector<Edge> edges;

    for (auto&& srcLayer: graph)
    {
        const unsigned int numOutputSlots = srcLayer->GetNumOutputSlots();
        for (unsigned int s = 0; s < numOutputSlots; ++s)
        {
            const armnn::IOutputSlot& outputSlot = srcLayer->GetOutputSlot(s);
            const unsigned int numConnections = outputSlot.GetNumConnections();
            for (unsigned int c = 0; c < numConnections; ++c)
            {
                auto inputSlot = armnn::PolymorphicDowncast<const armnn::InputSlot*>(outputSlot.GetConnection(c));
                edges.emplace_back(srcLayer, &inputSlot->GetOwningLayer());
            }
        }
    }

    return edges;
}

static void TestGraphAfterAddingCopyLayers(const armnn::Graph& graph, const armnn::Graph& origGraph)
{
    std::vector<Edge> origEdges = GetEdgeList(origGraph);
    std::vector<Edge> newEdges = GetEdgeList(graph);

    // Adding copy layers should not produce any duplicate edges.
    {
        std::vector<Edge> sortedNewEdges = newEdges;
        std::sort(sortedNewEdges.begin(), sortedNewEdges.end());

        auto last = std::unique(sortedNewEdges.begin(), sortedNewEdges.end());
        BOOST_CHECK_MESSAGE(last == sortedNewEdges.end(), "New graph contains duplicate edges!");
    }

    // Each new edge must be tested.
    while (!newEdges.empty())
    {
        const Edge edge = std::move(newEdges.back());
        newEdges.pop_back();

        // Edge present in the original graph?
        int originalEdge = -1;
        for (unsigned int i = 0; i < origEdges.size(); i++)
        {
            const Edge& origEdge = origEdges[i];
            if (origEdge.first->GetNameStr() == edge.first->GetNameStr() &&
                origEdge.second->GetNameStr() == edge.second->GetNameStr())
            {
                originalEdge = armnn::numeric_cast<int>(i);
            }
        }

        if (originalEdge != -1)
        {
            // Each vertex should correspond to a layer.
            const armnn::Layer* srcLayer = edge.first;
            const armnn::Layer* dstLayer = edge.second;
            BOOST_TEST(srcLayer);
            BOOST_TEST(dstLayer);

            // Both layers must have the same compute device.
            if (srcLayer && dstLayer)
            {
                BOOST_TEST((srcLayer->GetBackendId() == dstLayer->GetBackendId()));
            }

            // Marks edge in original graph as observed (by deleting it).
            origEdges.erase(origEdges.begin() + originalEdge);
        }
        else
        {
            // Edge did not exist in the original graph.
            // It must then be an edge connecting a layer and a copy layer.
            const armnn::Layer* srcLayer = edge.first;
            const armnn::Layer* dstLayer = edge.second;

            if (srcLayer == nullptr || dstLayer == nullptr)
            {
                BOOST_ERROR("At least one of the two ends of a new edge (" << edge.first << ", " << edge.second << ") "
                            "introduced after adding copy layers to a graph "
                            "correspond to a layer not known to the graph");
                continue;
            }

            // One and only one of the two layers referenced by the edge should be present in the original graph.
            const bool srcLayerInOrigGraph = GraphHasNamedLayer(origGraph, srcLayer->GetNameStr());
            const bool dstLayerInOrigGraph = GraphHasNamedLayer(origGraph, dstLayer->GetNameStr());

            if (srcLayerInOrigGraph == dstLayerInOrigGraph)
            {
                BOOST_ERROR("A new edge ("
                            << edge.first->GetName()
                            << ", "
                            << edge.second->GetName()
                            << ") introduced after adding copy "
                               "layers to a graph is invalid. One of the ends should be present in the original "
                               "graph and the other should not, but "
                            << (srcLayerInOrigGraph ? "both are" : "none are"));
                continue;
            }

            const armnn::Layer* copyLayer = srcLayerInOrigGraph ? dstLayer : srcLayer;
            const armnn::Layer* nonCopyLayer = srcLayerInOrigGraph ? srcLayer : dstLayer;

            // Finds all edges connecting the copy layer to other layers.
            std::vector<Edge> adjEdges;
            auto it = newEdges.begin();
            while (it != newEdges.end())
            {
                Edge& newEdge = *it;
                if (copyLayer == (srcLayerInOrigGraph ? newEdge.first : newEdge.second))
                {
                    adjEdges.push_back(newEdge);

                    // Since the adjacent edge is immediately tested below, there is no need to consider it afterwards.
                    it = newEdges.erase(it);
                }
                else
                {
                    it++;
                }
            }

            if (adjEdges.empty())
            {
                BOOST_ERROR("An edge connecting a layer and a copy layer exists, (" << edge.first << ", " <<
                            edge.second << "),  but no other edges connecting the copy layer '" << copyLayer->GetName()
                            << "' to other layers could be found");
                continue;
            }

            // Tests adjacent edges now.
            for (const Edge& adjEdge : adjEdges)
            {
                // The adjacent edge must connect the copy layer to another layer.
                const armnn::Layer* adjLayer = srcLayerInOrigGraph ? adjEdge.second : adjEdge.first;

                if (!adjLayer)
                {
                    BOOST_ERROR("An edge (" << adjEdge.first << ", " << adjEdge.second <<") is adjacent to an edge "
                                "connecting a layer and a copy layer, (" << edge.first << ", " << edge.second << "), "
                                "but the non-copy layer in the former does not correspond to a layer");
                    continue;
                }

                // Both layers must have different compute devices.
                BOOST_TEST((nonCopyLayer->GetBackendId() != adjLayer->GetBackendId()));

                // There must exist an edge connecting both layers directly in the original graph.
                {
                    const armnn::Layer* origEdgeSrc = srcLayerInOrigGraph ? nonCopyLayer : adjLayer;
                    const armnn::Layer* origEdgeDst = srcLayerInOrigGraph ? adjLayer : nonCopyLayer;

                    auto origEdgeIter = origEdges.begin();
                    for (; origEdgeIter != origEdges.end(); origEdgeIter++)
                    {
                        if (origEdgeIter->first->GetNameStr() == origEdgeSrc->GetNameStr() &&
                            origEdgeIter->second->GetNameStr() == origEdgeDst->GetNameStr())
                        {
                            break;
                        }
                    }

                    if (origEdgeIter != origEdges.end())
                    {
                        origEdges.erase(origEdgeIter);
                    }
                    else
                    {
                        BOOST_ERROR("An edge (" << adjEdge.first << ", " << adjEdge.second << ") is adjacent to an "
                            "edge connecting a layer and a copy layer, (" << edge.first << ", " << edge.second <<
                            "), but there is no edge connecting the layers in the original graph");
                    }
                }
            }
        }
    }

    BOOST_TEST(origEdges.empty(), "Not all of the edges in the original graph correspond to paths in the new graph");
}

struct CopyLayersFixture
{
    CopyLayersFixture()
    {
    }

    void InitialiseTestGraph()
    {
        using namespace armnn;
        using namespace std;

        Layer* const inputLayer = AddLayer<InputLayer>(0, "input");
        inputLayer->SetBackendId(Compute::CpuRef);

        Convolution2dDescriptor convolutionDefaults;
        Layer* const convLayer1 = AddLayer<Convolution2dLayer>(convolutionDefaults, "conv1");
        convLayer1->SetBackendId(Compute::CpuRef);

        inputLayer->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(0));

        Layer* const convLayer2 = AddLayer<Convolution2dLayer>(convolutionDefaults, "conv2");
        convLayer2->SetBackendId(Compute::CpuAcc);

        convLayer1->GetOutputSlot(0).Connect(convLayer2->GetInputSlot(0));

        armnn::OriginsDescriptor concatDefaults(2);
        Layer* const concatLayer = AddLayer<ConcatLayer>(concatDefaults, "concat");
        concatLayer->SetBackendId(armnn::Compute::CpuRef);

        convLayer1->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(0));
        convLayer2->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(1));

        armnn::ActivationDescriptor activationDefaults;
        Layer* const actLayer = AddLayer<ActivationLayer>(activationDefaults, "act");
        actLayer->SetBackendId(armnn::Compute::CpuRef);

        concatLayer->GetOutputSlot(0).Connect(actLayer->GetInputSlot(0));

        armnn::SoftmaxDescriptor softmaxDefaults;
        Layer* const softmaxLayer = AddLayer<SoftmaxLayer>(softmaxDefaults, "softmax");
        softmaxLayer->SetBackendId(armnn::Compute::CpuRef);

        actLayer->GetOutputSlot(0).Connect(softmaxLayer->GetInputSlot(0));

        Layer* const outputLayer = AddLayer<OutputLayer>(0, "output");
        outputLayer->SetBackendId(armnn::Compute::CpuAcc);

        softmaxLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

        // Set the memory strategies - for this test should be DirectCompatibility for same backends,
        // and CopyToTarget for different backends
        inputLayer->GetOutputSlot(0).SetEdgeStrategy(0, EdgeStrategy::DirectCompatibility);
        convLayer1->GetOutputSlot(0).SetEdgeStrategy(0, EdgeStrategy::CopyToTarget);
        convLayer1->GetOutputSlot(0).SetEdgeStrategy(1, EdgeStrategy::DirectCompatibility);
        convLayer2->GetOutputSlot(0).SetEdgeStrategy(0, EdgeStrategy::CopyToTarget);
        concatLayer->GetOutputSlot(0).SetEdgeStrategy(0, EdgeStrategy::DirectCompatibility);
        actLayer->GetOutputSlot(0).SetEdgeStrategy(0, EdgeStrategy::DirectCompatibility);
        softmaxLayer->GetOutputSlot(0).SetEdgeStrategy(0, EdgeStrategy::CopyToTarget);
    }

    armnn::TensorInfo m_TensorDesc;
    armnn::Graph m_Graph;
    std::map<armnn::BackendId, std::unique_ptr<armnn::IBackendInternal>> m_Backends;
    armnn::TensorHandleFactoryRegistry m_FactoryRegistry;

private:

    template <typename LayerType, typename... Args>
    LayerType* AddLayer(Args&&... args)
    {
        LayerType* const layer = m_Graph.AddLayer<LayerType>(std::forward<Args>(args)...);

        for (auto slot = layer->BeginOutputSlots(); slot != layer->EndOutputSlots(); ++slot)
        {
            slot->SetTensorInfo(m_TensorDesc);
        }

        return layer;
    };
};

BOOST_FIXTURE_TEST_CASE(AddCopyLayers, CopyLayersFixture)
{
    InitialiseTestGraph();
    const armnn::Graph origGraph(m_Graph);
    m_Graph.AddCompatibilityLayers(m_Backends, m_FactoryRegistry);

    TestGraphAfterAddingCopyLayers(m_Graph, origGraph);
}

BOOST_FIXTURE_TEST_CASE(AddCopyLayersSeveralTimes, CopyLayersFixture)
{
    InitialiseTestGraph();
    m_Graph.AddCompatibilityLayers(m_Backends, m_FactoryRegistry);

    // Calling AddCompatibilityLayers() several times should not change the connections.
    const std::vector<Edge> edges = GetEdgeList(m_Graph);
    for (int i = 0; i < 4; ++i)
    {
        m_Graph.AddCompatibilityLayers(m_Backends, m_FactoryRegistry);
        const std::vector<Edge> otherEdges = GetEdgeList(m_Graph);
        BOOST_TEST((edges == otherEdges));
    }
}

BOOST_FIXTURE_TEST_CASE(CopyLayersAddedBetweenSameLayersHaveDifferentNames, CopyLayersFixture)
{
    armnn::Graph graph;

    armnn::InputLayer* const inputLayer = graph.AddLayer<armnn::InputLayer>(0, "input");
    inputLayer->SetBackendId(armnn::Compute::CpuRef);

    armnn::ViewsDescriptor splitterDesc(2);
    armnn::SplitterLayer* const splitterLayer = graph.AddLayer<armnn::SplitterLayer>(splitterDesc, "splitter");
    splitterLayer->SetBackendId(armnn::Compute::GpuAcc);

    armnn::AdditionLayer* const additionLayer = graph.AddLayer<armnn::AdditionLayer>("addition");
    additionLayer->SetBackendId(armnn::Compute::CpuRef);

    armnn::OutputLayer* const outputLayer = graph.AddLayer<armnn::OutputLayer>(0, "output");
    outputLayer->SetBackendId(armnn::Compute::CpuRef);

    inputLayer->GetOutputSlot(0).Connect(splitterLayer->GetInputSlot(0));
    splitterLayer->GetOutputSlot(0).Connect(additionLayer->GetInputSlot(0));
    splitterLayer->GetOutputSlot(1).Connect(additionLayer->GetInputSlot(1));
    additionLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetEdgeStrategy(0, armnn::EdgeStrategy::DirectCompatibility);
    splitterLayer->GetOutputSlot(0).SetEdgeStrategy(0, armnn::EdgeStrategy::CopyToTarget);
    splitterLayer->GetOutputSlot(1).SetEdgeStrategy(0, armnn::EdgeStrategy::CopyToTarget);
    additionLayer->GetOutputSlot(0).SetEdgeStrategy(0, armnn::EdgeStrategy::DirectCompatibility);

    graph.AddCompatibilityLayers(m_Backends, m_FactoryRegistry);

    std::vector<Edge> edges = GetEdgeList(graph);
    BOOST_CHECK(edges.size() == 6u);
    std::sort(edges.begin(), edges.end());
    auto last = std::unique(edges.begin(), edges.end());
    BOOST_CHECK_MESSAGE(last == edges.end(), "Found duplicated edges after AddCompatibilityLayers()");
}

BOOST_AUTO_TEST_CASE(DuplicateLayerNames)
{
    armnn::Graph graph;

    armnn::InputLayer* const inputLayer = graph.AddLayer<armnn::InputLayer>(0, "layer");
    inputLayer->SetBackendId(armnn::Compute::CpuRef);

    armnn::OutputLayer* const outputLayer = graph.AddLayer<armnn::OutputLayer>(0, "layer");
    outputLayer->SetBackendId(armnn::Compute::CpuRef);

    inputLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    auto it = graph.TopologicalSort().begin();
    BOOST_TEST(((*it)->GetType() == armnn::LayerType::Input));
    BOOST_TEST(((*std::next(it))->GetType() == armnn::LayerType::Output));
}

BOOST_AUTO_TEST_SUITE_END()
