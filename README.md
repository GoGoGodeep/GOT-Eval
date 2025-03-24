# GOT-Eval

### 基于源码，增加部分代码用于更好的显示模型的推理结果  

process.py：对原图进行裁剪，获取ocr区域  
paint_result.py：根据ocr的result.txt文件在图片上绘制结果  
run_ocr_2.0.py（GOT-OCR-2.0-master/GOT/demo/run_ocr_2.0.py）：增加解码每个token并关联置信度，将结果保存到txt文件的功能；修改__main__函数  
GOT_ocr_2_0.py（GOT-OCR-2.0-master/GOT/model/GOT_ocr_2_0.py）：增加计算并缓存置信度的功能（267-269行）
