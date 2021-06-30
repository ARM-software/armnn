//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "AddBroadcastReshapeLayer.hpp"
#include "AddDebug.hpp"
#include "ConvertConstants.hpp"
#include "ConvertFp32NetworkToBf16.hpp"
#include "ConvertFp32NetworkToFp16.hpp"
#include "FoldPadIntoLayer2d.hpp"
#include "FuseBatchNorm.hpp"
#include "MovePermuteUp.hpp"
#include "MoveTransposeUp.hpp"
#include "OptimizeConsecutiveReshapes.hpp"
#include "OptimizeInverseConversions.hpp"
#include "OptimizeInversePermutes.hpp"
#include "PermuteAsReshape.hpp"
#include "PermuteAndBatchToSpaceAsDepthToSpace.hpp"
#include "RedirectMembersToConstantInputs.hpp"
#include "SquashEqualSiblings.hpp"
#include "TransposeAsReshape.hpp"