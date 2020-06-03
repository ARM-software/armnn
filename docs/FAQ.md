Frequently asked questions
==========================

These are issues that have been commonly seen when using ArmNN.

Problems seen when trying to build armnn and ComputeLibrary obtained from GitHub
-----------------------------------------------------------------------------

Some users have encountered difficulties when attempting to build armnn and ComputeLibrary obtained from GitHub. The build generally fails reporting missing dependencies or fields in aclCommon, backendsCommon, cl or neon. These errors can look like this:

error: ‘HARD_SWISH’ is not a member of ‘AclActivationFunction {aka arm_compute::ActivationLayerInfo::ActivationFunction}’

The most common reason for these errors are a mismatch between armnn and clframework revisions. For any version of ArmNN the coresponding version of ComputeLibrary is detailed in scripts/get_compute_library.sh as DEFAULT_CLFRAMEWORKREVISION

On *nix like systems running this script will checkout ComputeLibrary, with the current default SHA, into ../../clframework/ relative to the location of the script.

Segmentation fault following a failed call to armnn::Optimize using CpuRef backend.
---------------------------------------------------------

In some error scenarios of calls to armnn::Optimize a null pointer may be returned. This contravenes the function documentation however, it can happen. Users are advised to check the value returned from the function as a precaution.

If you encounter this problem and are able to isolate it consider contributing a solution.

Adding or removing -Dxxx options when building ArmNN does not always work.
---------------------------------------------------------

ArmNN uses CMake for build configuration. CMake uses a cumulative cache of user options. That is, setting a value once on a cmake command line will be persisted until either you explicitly change the value or delete the cache. To delete the cache in ArmNN you must delete the build directory.
