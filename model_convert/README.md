<!-- 1、paddle转onnx： -->
安装paddlepaddle、paddle2onnx
<!-- 检测： -->
paddle2onnx --model_dir ./PP-OCRv5_mobile_det_infer --model_filename inference.json --params_filename inference.pdiparams --save_file ./det_mobile.onnx --opset_version 11 --enable_onnx_checker True
<!-- 识别： -->
paddle2onnx --model_dir ./PP-OCRv5_mobile_rec_infer --model_filename inference.json --params_filename inference.pdiparams --save_file ./rec_mobile.onnx --opset_version 11 --enable_onnx_checker True
<!-- 分类： -->
paddle2onnx --model_dir ./PP-LCNet_x0_25_textline_ori_infer --model_filename inference.json --params_filename inference.pdiparams --save_file ./cls_mobile.onnx --opset_version 11 --enable_onnx_checker True

<!-- 2、动态onnx转静态 -->
onnxsim det_mobile.onnx  det_mobile_sim_static.onnx --overwrite-input-shape=1,3,960,960
onnxsim rec_mobile.onnx  rec_mobile_sim_static.onnx --overwrite-input-shape=1,3,48,320
onnxsim cls_mobile.onnx  cls_mobile_sim_static.onnx --overwrite-input-shape=1,3,80,160

<!-- 3、模型转换 -->
pulsar2 build --config ./det.json
pulsar2 build --config ./rec.json
pulsar2 build --config ./cls.json

- 使用 `Pulsar2 build` 命令集进行模型转换（PTQ 量化、编译），更详细的使用说明请参考 [AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)