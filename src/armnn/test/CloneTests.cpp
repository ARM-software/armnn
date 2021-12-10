//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <Graph.hpp>
#include <Layer.hpp>

#include <armnn/TypesUtils.hpp>
#include <armnn/Exceptions.hpp>
#include <armnn/Optional.hpp>
#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <backendsCommon/WorkloadFactoryBase.hpp>

#include <doctest/doctest.h>

namespace {

const armnn::BackendId& GetCloneIdStatic()
{
    static const armnn::BackendId s_Id{"Tests"};
    return s_Id;
}

template <typename T>
void DeleteAsType(const void* const blob)
{
    delete static_cast<const T*>(blob);
}

class TestWorkloadFactory : public armnn::WorkloadFactoryBase
{
public:

    TestWorkloadFactory(void* ptr)
        : m_Ptr(ptr)
    {}

    const armnn::BackendId& GetBackendId() const override
    {
        return GetCloneIdStatic();
    }

    std::unique_ptr<armnn::IWorkload> CreatePreCompiled(const armnn::PreCompiledQueueDescriptor& descriptor,
                                                        const armnn::WorkloadInfo&) const override
    {
        CHECK(descriptor.m_PreCompiledObject == m_Ptr);
        return nullptr;
    }

    mutable void* m_Ptr;
};

TEST_SUITE("CloneTests")
{

TEST_CASE ("PreCompiledLayerClonePreservesObject")
{
    armnn::Graph graph1;
    armnn::Graph graph2;

    armnn::PreCompiledDescriptor descriptor(0u, 0u);

    armnn::Layer* const preCompiledLayer = graph1.AddLayer<armnn::PreCompiledLayer>(descriptor, "preCompiled");
    armnn::PreCompiledLayer* layer = armnn::PolymorphicDowncast<armnn::PreCompiledLayer*>(preCompiledLayer);
    std::unique_ptr<std::string> payload = std::make_unique<std::string>("Hello");

    armnn::PreCompiledObjectPtr payloadObject(payload.release(), DeleteAsType<std::string>);
    TestWorkloadFactory factory(payloadObject.get());

    layer->SetPreCompiledObject(std::move(payloadObject));
    layer->CreateWorkload(factory);

    armnn::PreCompiledLayer* clone = layer->Clone(graph2);
    CHECK(std::strcmp(clone->GetName(), "preCompiled") == 0);
    clone->CreateWorkload(factory);
}

TEST_CASE ("PreCompiledLayerCloneNoObject")
{
    armnn::Graph graph1;

    armnn::Graph graph2;

    armnn::PreCompiledDescriptor descriptor(0u, 0u);

    armnn::Layer* const preCompiledLayer = graph1.AddLayer<armnn::PreCompiledLayer>(descriptor, "preCompiled");
    armnn::PreCompiledLayer* layer = armnn::PolymorphicDowncast<armnn::PreCompiledLayer*>(preCompiledLayer);

    TestWorkloadFactory factory(nullptr);
    layer->CreateWorkload(factory);

    armnn::PreCompiledLayer* clone = layer->Clone(graph2);
    CHECK(std::strcmp(clone->GetName(), "preCompiled") == 0);
    clone->CreateWorkload(factory);
}

}

} // end anonymous namespace
