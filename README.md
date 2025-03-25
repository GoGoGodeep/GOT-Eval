# 🚀 GOT-Eval  

基于GOT-OCR 2.0的增强版本，优化了模型推理结果的可视化效果  

## ✨修改内容  

在原版代码[Ucas-HaoranWei/GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)的基础上，新增了以下功能以实现更直观的结果展示：  

1. **process.py**  
   - 对原始图像进行裁剪，提取OCR识别区域  

2. **paint_result.py**  
   - 根据result.txt文件中的识别结果，在图像上绘制OCR文本  

3. **run_ocr_2.0.py**（路径：`GOT-OCR-2.0-master/GOT/demo/run_ocr_2.0.py`）  
   - 新增功能：解码每个token并关联置信度分数  
   - 将识别结果保存至txt文件  
   - 修改了`__main__`主函数  

4. **GOT_ocr_2_0.py**（路径：`GOT-OCR-2.0-master/GOT/model/GOT_ocr_2_0.py`）  
   - 新增置信度计算与缓存功能（267-269行）  

## 🌟使用说明  

1. 使用`process.py`预处理图像，提取OCR识别区域  
2. 运行OCR推理，生成结果文件  
3. 通过`paint_result.py`在图像上可视化识别结果  

本版本通过显示置信度分数和可视化验证功能，能更直观地评估模型性能。
