from PIL import Image
import os

# 设置路径和裁剪坐标
ori_dir = '/home/zhoukefan/GOT-OCR2.0/imgs/16/ori'
cut_dir = '/home/zhoukefan/GOT-OCR2.0/imgs/16/cut'
x1, y1 = 468, 338  # 左上角坐标
x2, y2 = 1026, 429  # 右下角坐标

# 创建输出目录
os.makedirs(cut_dir, exist_ok=True)

# 支持的图片格式
supported_exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

# 处理每张图片
for filename in os.listdir(ori_dir):
    # 检查文件扩展名
    ext = os.path.splitext(filename)[1].lower()
    if ext not in supported_exts:
        continue
    
    # 处理文件路径
    img_path = os.path.join(ori_dir, filename)
    output_path = os.path.join(cut_dir, filename)
    
    try:
        # 打开并裁剪图片
        with Image.open(img_path) as img:
            cropped = img.crop((x1, y1, x2, y2))
            cropped.save(output_path)
            print(f'成功处理: {filename}')
    except Exception as e:
        print(f'处理 {filename} 时出错: {str(e)}')

print('所有图片处理完成！')
