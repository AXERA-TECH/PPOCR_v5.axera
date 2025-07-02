# 模型转换

## 创建虚拟环境

```
conda create -n PPOCR python=3.12 -y
conda activate PPOCR
```
建议创建python虚拟环境，参照（https://github.com/PaddlePaddle/PaddleOCR）安装好paddlepaddle、paddleocr、paddle2onnx，用以下命令将mobile模型导出为对应的onnx模型：

## 导出模型（Paddle -> ONNX）
### 检测：
```
paddle2onnx --model_dir ./PP-OCRv5_mobile_det_infer --model_filename inference.json --params_filename inference.pdiparams --save_file ./det_mobile.onnx --opset_version 11 --enable_onnx_checker True
```

### 识别：
```
paddle2onnx --model_dir ./PP-OCRv5_mobile_rec_infer --model_filename inference.json --params_filename inference.pdiparams --save_file ./rec_mobile.onnx --opset_version 11 --enable_onnx_checker True
```

### 分类：
```
paddle2onnx --model_dir ./PP-LCNet_x0_25_textline_ori_infer --model_filename inference.json --params_filename inference.pdiparams --save_file ./cls_mobile.onnx --opset_version 11 --enable_onnx_checker True
```
导出成功后会生成'det_mobile.onnx'、'rec_mobile.onnx'、'cls_mobile.onnx'三个模型.

## 动态onnx转静态
```
onnxsim det_mobile.onnx  det_mobile_sim_static.onnx --overwrite-input-shape=1,3,960,960
onnxsim rec_mobile.onnx  rec_mobile_sim_static.onnx --overwrite-input-shape=1,3,48,320
onnxsim cls_mobile.onnx  cls_mobile_sim_static.onnx --overwrite-input-shape=1,3,80,160
```

## 转换模型（ONNX -> Axera）
使用模型转换工具 `Pulsar2` 将 ONNX 模型转换成适用于 Axera 的 NPU 运行的模型文件格式 `.axmodel`，通常情况下需要经过以下两个步骤：

- 生成适用于该模型的 PTQ 量化校准数据集
- 使用 `Pulsar2 build` 命令集进行模型转换（PTQ 量化、编译），更详细的使用说明请参考 [AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)

### 下载量化数据集
```
bash download_dataset.sh
```
这个模型的输入是单张图片，比较简单，这里我们直接下载打包好的图片数据  

### 模型转换

#### 修改配置文件
 
检查`config.json` 中 `calibration_dataset` 字段，将该字段配置的路径改为上一步下载的量化数据集存放路径  

#### Pulsar2 build

参考命令如下：

```
pulsar2 build --config ./det.json
pulsar2 build --config ./rec.json
pulsar2 build --config ./cls.json
```
