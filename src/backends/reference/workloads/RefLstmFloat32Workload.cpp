//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefLstmFloat32Workload.hpp"

namespace armnn
{

void RefLstmFloat32Workload::Execute() const
{
    throw armnn::Exception("No implementation of Lstm in the Ref backend!");
}

} //namespace armnn
