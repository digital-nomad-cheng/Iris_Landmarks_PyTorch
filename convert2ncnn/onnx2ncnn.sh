python -m onnxsim iris_lnet.onnx iris_lnet.onnx
echo "Finished simplified"
/home/idealabs/Libs/ncnn2/build/tools/onnx/onnx2ncnn iris_lnet.onnx iris_lnet.param iris_lnet.bin
