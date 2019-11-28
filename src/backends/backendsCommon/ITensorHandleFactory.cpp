//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/backends/ITensorHandleFactory.hpp>

namespace armnn
{

const ITensorHandleFactory::FactoryId ITensorHandleFactory::LegacyFactoryId = "armnn_legacy_factory";
const ITensorHandleFactory::FactoryId ITensorHandleFactory::DeferredFactoryId = "armnn_deferred_factory";

} // namespace armnn
