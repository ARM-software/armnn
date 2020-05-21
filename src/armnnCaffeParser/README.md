# Arm NN Caffe parser

`armnnCaffeParser` is a library for loading neural networks defined in Caffe protobuf files into the Arm NN runtime.

For more information about the Caffe layers that are supported, and the networks that have been tested, see [CaffeSupport.md](./CaffeSupport.md).

Please note that certain deprecated Caffe features are not supported by the armnnCaffeParser. If you think that Arm NN should be able to load your model according to the list of supported layers in [CaffeSupport.md](./CaffeSupport.md), but you are getting strange error messages, then try upgrading your model to the latest format using Caffe, either by saving it to a new file or using the upgrade utilities in `caffe/tools`.
