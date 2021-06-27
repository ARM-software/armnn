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
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <backendsCommon/TensorHandle.hpp>
#include <backendsCommon/WorkloadFactoryBase.hpp>

#include <doctest/doctest.h>

namespace {

const armnn::BackendId& GetCloneIdStatic()
{
    static const armnn::BackendId s_Id{"Tests"};
    return s_Id;
}

class TestWorkloadFactory : public armnn::WorkloadFactoryBase
{
public:

    TestWorkloadFactory()
        : m_Ptr(nullptr)
    {}

    const armnn::BackendId& GetBackendId() const override
    {
        return GetCloneIdStatic();
    }

    std::unique_ptr<armnn::IWorkload> CreatePreCompiled(const armnn::PreCompiledQueueDescriptor& descriptor,
                                                        const armnn::WorkloadInfo&) const override
    {
        if (m_Ptr)
        {
            CHECK(descriptor.m_PreCompiledObject == m_Ptr);
        }
        else
        {
            m_Ptr = descriptor.m_PreCompiledObject;
        }
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

    armnn::PreCompiledObjectPtr payloadObject;
    TestWorkloadFactory factory;

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

    TestWorkloadFactory factory;
    layer->CreateWorkload(factory);

    armnn::PreCompiledLayer* clone = layer->Clone(graph2);
    CHECK(std::strcmp(clone->GetName(), "preCompiled") == 0);
    clone->CreateWorkload(factory);
}

}

} // end anonymous namespace
