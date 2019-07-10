//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/BackendId.hpp>
#include <armnn/ILayerSupport.hpp>

namespace armnn
{

/// Convenience function to retrieve the ILayerSupport for a backend
std::shared_ptr<ILayerSupport> GetILayerSupportByBackendId(const armnn::BackendId& backend);

}
