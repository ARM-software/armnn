//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "ConvertConstants.hpp"
#include "OptimizeInversePermutes.hpp"
#include "PermuteAsReshape.hpp"
#include "OptimizeConsecutiveReshapes.hpp"
#include "SquashEqualSiblings.hpp"
#include "MovePermuteUp.hpp"
#include "OptimizeInverseConversions.hpp"
#include "ConvertFp32NetworkToFp16.hpp"
