//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <doctest/doctest.h>

#include <armnn/LayerVisitorBase.hpp>

#include <armnn/backends/IBackendContext.hpp>
#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/IMemoryManager.hpp>
#include <armnn/backends/ITensorHandleFactory.hpp>
#include <backendsCommon/TensorHandleFactoryRegistry.hpp>

#include <optimizations/Optimization.hpp>

#include <Network.hpp>

#include <armnn/utility/IgnoreUnused.hpp>

#include <vector>
#include <string>


using namespace armnn;

class TestMemMgr : public IMemoryManager
{
public:
    TestMemMgr() = default;

    void Acquire() override {}
    void Release() override {}
};

class TestFactory1 : public ITensorHandleFactory
{
public:
    TestFactory1(std::weak_ptr<IMemoryManager> mgr, ITensorHandleFactory::FactoryId id)
        : m_Id(id)
        , m_MemMgr(mgr)
    {}

    std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                         TensorShape const& subTensorShape,
                                                         unsigned int const* subTensorOrigin) const override
    {
        IgnoreUnused(parent, subTensorShape, subTensorOrigin);
        return nullptr;
    }

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo) const override
    {
        IgnoreUnused(tensorInfo);
        return nullptr;
    }

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      DataLayout dataLayout) const override
    {
        IgnoreUnused(tensorInfo, dataLayout);
        return nullptr;
    }

    const FactoryId& GetId() const override { return m_Id; }

    bool SupportsSubTensors() const override { return true; }

    MemorySourceFlags GetExportFlags() const override { return 1; }

private:
    FactoryId m_Id = "UninitializedId";

    std::weak_ptr<IMemoryManager> m_MemMgr;
};

class TestFactoryImport : public ITensorHandleFactory
{
public:
    TestFactoryImport(std::weak_ptr<IMemoryManager> mgr, ITensorHandleFactory::FactoryId id)
        : m_Id(id)
        , m_MemMgr(mgr)
    {}

    std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                         TensorShape const& subTensorShape,
                                                         unsigned int const* subTensorOrigin) const override
    {
        IgnoreUnused(parent, subTensorShape, subTensorOrigin);
        return nullptr;
    }

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo) const override
    {
        IgnoreUnused(tensorInfo);
        return nullptr;
    }

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      DataLayout dataLayout) const override
    {
        IgnoreUnused(tensorInfo, dataLayout);
        return nullptr;
    }

    const FactoryId& GetId() const override { return m_Id; }

    bool SupportsSubTensors() const override { return true; }

    MemorySourceFlags GetImportFlags() const override { return 1; }

private:
    FactoryId m_Id = "ImporterId";

    std::weak_ptr<IMemoryManager> m_MemMgr;
};

class TestBackendA : public IBackendInternal
{
public:
    TestBackendA() = default;

    const BackendId& GetId() const override { return m_Id; }

    IWorkloadFactoryPtr CreateWorkloadFactory(const IMemoryManagerSharedPtr& memoryManager = nullptr) const override
    {
        IgnoreUnused(memoryManager);
        return IWorkloadFactoryPtr{};
    }

    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override
    {
        return ILayerSupportSharedPtr{};
    }

    std::vector<ITensorHandleFactory::FactoryId> GetHandleFactoryPreferences() const override
    {
        return std::vector<ITensorHandleFactory::FactoryId>
        {
            "TestHandleFactoryA1",
            "TestHandleFactoryA2",
            "TestHandleFactoryB1",
            "TestHandleFactoryD1"
        };
    }

    void RegisterTensorHandleFactories(TensorHandleFactoryRegistry& registry) override
    {
        auto mgr = std::make_shared<TestMemMgr>();

        registry.RegisterMemoryManager(mgr);
        registry.RegisterFactory(std::make_unique<TestFactory1>(mgr, "TestHandleFactoryA1"));
        registry.RegisterFactory(std::make_unique<TestFactory1>(mgr, "TestHandleFactoryA2"));
    }

private:
    BackendId m_Id = "BackendA";
};

class TestBackendB : public IBackendInternal
{
public:
    TestBackendB() = default;

    const BackendId& GetId() const override { return m_Id; }

    IWorkloadFactoryPtr CreateWorkloadFactory(const IMemoryManagerSharedPtr& memoryManager = nullptr) const override
    {
        IgnoreUnused(memoryManager);
        return IWorkloadFactoryPtr{};
    }

    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override
    {
        return ILayerSupportSharedPtr{};
    }

    std::vector<ITensorHandleFactory::FactoryId> GetHandleFactoryPreferences() const override
    {
        return std::vector<ITensorHandleFactory::FactoryId>
        {
            "TestHandleFactoryB1"
        };
    }

    void RegisterTensorHandleFactories(TensorHandleFactoryRegistry& registry) override
    {
        auto mgr = std::make_shared<TestMemMgr>();

        registry.RegisterMemoryManager(mgr);
        registry.RegisterFactory(std::make_unique<TestFactory1>(mgr, "TestHandleFactoryB1"));
    }

private:
    BackendId m_Id = "BackendB";
};

class TestBackendC : public IBackendInternal
{
public:
    TestBackendC() = default;

    const BackendId& GetId() const override { return m_Id; }

    IWorkloadFactoryPtr CreateWorkloadFactory(const IMemoryManagerSharedPtr& memoryManager = nullptr) const override
    {
        IgnoreUnused(memoryManager);
        return IWorkloadFactoryPtr{};
    }

    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override
    {
        return ILayerSupportSharedPtr{};
    }

    std::vector<ITensorHandleFactory::FactoryId> GetHandleFactoryPreferences() const override
    {
        return std::vector<ITensorHandleFactory::FactoryId>{
            "TestHandleFactoryC1"
        };
    }

    void RegisterTensorHandleFactories(TensorHandleFactoryRegistry& registry) override
    {
        auto mgr = std::make_shared<TestMemMgr>();

        registry.RegisterMemoryManager(mgr);
        registry.RegisterFactory(std::make_unique<TestFactory1>(mgr, "TestHandleFactoryC1"));
    }

private:
    BackendId m_Id = "BackendC";
};

class TestBackendD : public IBackendInternal
{
public:
    TestBackendD() = default;

    const BackendId& GetId() const override { return m_Id; }

    IWorkloadFactoryPtr CreateWorkloadFactory(const IMemoryManagerSharedPtr& memoryManager = nullptr) const override
    {
        IgnoreUnused(memoryManager);
        return IWorkloadFactoryPtr{};
    }

    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override
    {
        return ILayerSupportSharedPtr{};
    }

    std::vector<ITensorHandleFactory::FactoryId> GetHandleFactoryPreferences() const override
    {
        return std::vector<ITensorHandleFactory::FactoryId>{
            "TestHandleFactoryD1",
        };
    }

    void RegisterTensorHandleFactories(TensorHandleFactoryRegistry& registry) override
    {
        auto mgr = std::make_shared<TestMemMgr>();

        registry.RegisterMemoryManager(mgr);
        registry.RegisterFactory(std::make_unique<TestFactoryImport>(mgr, "TestHandleFactoryD1"));
    }

private:
    BackendId m_Id = "BackendD";
};


TEST_SUITE("TensorHandle")
{
TEST_CASE("RegisterFactories")
{
    TestBackendA backendA;
    TestBackendB backendB;

    CHECK(backendA.GetHandleFactoryPreferences()[0] == "TestHandleFactoryA1");
    CHECK(backendA.GetHandleFactoryPreferences()[1] == "TestHandleFactoryA2");
    CHECK(backendA.GetHandleFactoryPreferences()[2] == "TestHandleFactoryB1");
    CHECK(backendA.GetHandleFactoryPreferences()[3] == "TestHandleFactoryD1");

    TensorHandleFactoryRegistry registry;
    backendA.RegisterTensorHandleFactories(registry);
    backendB.RegisterTensorHandleFactories(registry);

    CHECK((registry.GetFactory("Non-existing Backend") == nullptr));
    CHECK((registry.GetFactory("TestHandleFactoryA1") != nullptr));
    CHECK((registry.GetFactory("TestHandleFactoryA2") != nullptr));
    CHECK((registry.GetFactory("TestHandleFactoryB1") != nullptr));
}

TEST_CASE("TensorHandleSelectionStrategy")
{
    auto backendA = std::make_unique<TestBackendA>();
    auto backendB = std::make_unique<TestBackendB>();
    auto backendC = std::make_unique<TestBackendC>();
    auto backendD = std::make_unique<TestBackendD>();

    TensorHandleFactoryRegistry registry;
    backendA->RegisterTensorHandleFactories(registry);
    backendB->RegisterTensorHandleFactories(registry);
    backendC->RegisterTensorHandleFactories(registry);
    backendD->RegisterTensorHandleFactories(registry);

    BackendsMap backends;
    backends["BackendA"] = std::move(backendA);
    backends["BackendB"] = std::move(backendB);
    backends["BackendC"] = std::move(backendC);
    backends["BackendD"] = std::move(backendD);

    armnn::Graph graph;

    armnn::InputLayer* const inputLayer = graph.AddLayer<armnn::InputLayer>(0, "input");
    inputLayer->SetBackendId("BackendA");

    armnn::SoftmaxDescriptor smDesc;
    armnn::SoftmaxLayer* const softmaxLayer1 = graph.AddLayer<armnn::SoftmaxLayer>(smDesc, "softmax1");
    softmaxLayer1->SetBackendId("BackendA");

    armnn::SoftmaxLayer* const softmaxLayer2 = graph.AddLayer<armnn::SoftmaxLayer>(smDesc, "softmax2");
    softmaxLayer2->SetBackendId("BackendB");

    armnn::SoftmaxLayer* const softmaxLayer3 = graph.AddLayer<armnn::SoftmaxLayer>(smDesc, "softmax3");
    softmaxLayer3->SetBackendId("BackendC");

    armnn::SoftmaxLayer* const softmaxLayer4 = graph.AddLayer<armnn::SoftmaxLayer>(smDesc, "softmax4");
    softmaxLayer4->SetBackendId("BackendD");

    armnn::OutputLayer* const outputLayer = graph.AddLayer<armnn::OutputLayer>(0, "output");
    outputLayer->SetBackendId("BackendA");

    inputLayer->GetOutputSlot(0).Connect(softmaxLayer1->GetInputSlot(0));
    softmaxLayer1->GetOutputSlot(0).Connect(softmaxLayer2->GetInputSlot(0));
    softmaxLayer2->GetOutputSlot(0).Connect(softmaxLayer3->GetInputSlot(0));
    softmaxLayer3->GetOutputSlot(0).Connect(softmaxLayer4->GetInputSlot(0));
    softmaxLayer4->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    graph.TopologicalSort();

    std::vector<std::string> errors;
    auto result = SelectTensorHandleStrategy(graph, backends, registry, true, errors);

    CHECK(result.m_Error == false);
    CHECK(result.m_Warning == false);

    OutputSlot& inputLayerOut = inputLayer->GetOutputSlot(0);
    OutputSlot& softmaxLayer1Out = softmaxLayer1->GetOutputSlot(0);
    OutputSlot& softmaxLayer2Out = softmaxLayer2->GetOutputSlot(0);
    OutputSlot& softmaxLayer3Out = softmaxLayer3->GetOutputSlot(0);
    OutputSlot& softmaxLayer4Out = softmaxLayer4->GetOutputSlot(0);

    // Check that the correct factory was selected
    CHECK(inputLayerOut.GetTensorHandleFactoryId() == "TestHandleFactoryD1");
    CHECK(softmaxLayer1Out.GetTensorHandleFactoryId() == "TestHandleFactoryB1");
    CHECK(softmaxLayer2Out.GetTensorHandleFactoryId() == "TestHandleFactoryB1");
    CHECK(softmaxLayer3Out.GetTensorHandleFactoryId() == "TestHandleFactoryC1");
    CHECK(softmaxLayer4Out.GetTensorHandleFactoryId() == "TestHandleFactoryD1");

    // Check that the correct strategy was selected
    CHECK((inputLayerOut.GetEdgeStrategyForConnection(0) == EdgeStrategy::DirectCompatibility));
    CHECK((softmaxLayer1Out.GetEdgeStrategyForConnection(0) == EdgeStrategy::DirectCompatibility));
    CHECK((softmaxLayer2Out.GetEdgeStrategyForConnection(0) == EdgeStrategy::CopyToTarget));
    CHECK((softmaxLayer3Out.GetEdgeStrategyForConnection(0) == EdgeStrategy::ExportToTarget));
    CHECK((softmaxLayer4Out.GetEdgeStrategyForConnection(0) == EdgeStrategy::DirectCompatibility));

    graph.AddCompatibilityLayers(backends, registry);

    // Test for copy layers
    int copyCount= 0;
    graph.ForEachLayer([&copyCount](Layer* layer)
    {
        if (layer->GetType() == LayerType::MemCopy)
        {
            copyCount++;
        }
    });
    CHECK(copyCount == 1);

    // Test for import layers
    int importCount= 0;
    graph.ForEachLayer([&importCount](Layer *layer)
    {
        if (layer->GetType() == LayerType::MemImport)
        {
            importCount++;
        }
    });
    CHECK(importCount == 1);
}

TEST_CASE("RegisterCopyAndImportFactoryPairTest")
{
    TensorHandleFactoryRegistry registry;
    ITensorHandleFactory::FactoryId copyId = "CopyFactoryId";
    ITensorHandleFactory::FactoryId importId = "ImportFactoryId";
    registry.RegisterCopyAndImportFactoryPair(copyId, importId);

    // Get mathing import factory id correctly
    CHECK((registry.GetMatchingImportFactoryId(copyId) == importId));

    // Return empty id when Invalid Id is given
    CHECK((registry.GetMatchingImportFactoryId("InvalidFactoryId") == ""));
}

}
