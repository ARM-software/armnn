//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

namespace armnn
{

struct LstmInputParams;
struct QuantizedLstmInputParams;

namespace experimental
{

class IAsyncNetwork;

} // end experimental namespace

class INetwork;
class IOptimizedNetwork;
class Graph;
class IInputSlot;
class IOutputSlot;
class IConnectableLayer;
class IDataLayer;

} // end armnn namespace
