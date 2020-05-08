Frequently asked questions
==========================

These are issues that have been commonly seen when using ArmNN.

Segmentation fault following a failed call to armnn::Optimize using CpuRef backend.
---------------------------------------------------------

In some error scenarios of calls to armnn::Optimize a null pointer may be
returned. This contravenes the function documentation however, it can
happen. Users are advised to check the value returned from the function as a
precaution.

If you encounter this problem and are able to isolate it consider contributing
a solution.
