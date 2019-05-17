//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/OptimizationViews.hpp>

namespace armnn
{

class MockBackend : public IBackendInternal
{
public:
    MockBackend()  = default;
    ~MockBackend() = default;

    static const BackendId& GetIdStatic();
    const BackendId& GetId() const override { return GetIdStatic(); }

    IBackendInternal::IMemoryManagerUniquePtr CreateMemoryManager() const override;

    IBackendInternal::IWorkloadFactoryPtr CreateWorkloadFactory(
        const IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr) const override;

    IBackendInternal::IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const override;

    IBackendInternal::Optimizations GetOptimizations() const override;
    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override;

    OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph) const override;
};

} // namespace armnn
