//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "StringMapping.hpp"

namespace armnn
{

const StringMapping& StringMapping::Instance()
{
    static StringMapping instance;
    return instance;
}

} // armnn
