from PIL import Image, ImageDraw, ImageFont
import re
import os

def process_ocr_results(txt_path, output_dir, font_size=20, font_path="/home/zhoukefan/SimHei.ttf", max_per_line=5):
    """
    扩展图片画布追加OCR信息
    参数:
        txt_path: OCR结果文本文件路径
        output_dir: 输出目录
        font_size: 字体大小
    """
    # 读取文本内容
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 分割处理块
    blocks = re.split(r'\n\s*\n', content.strip())
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 4:
            continue

        try:
            # 解析数据
            img_path = lines[0]

            # 新增标签过滤逻辑
            inference = (lines[1].split("：", 1)[1]).strip()
            inference = inference[len("<ref>0</ref>"):].strip()

            # 处理置信度显示优化
            conf_values = list(map(float, lines[3].split("：", 1)[1].split(',')))[3:]
            # print(conf_values)
            avg_confidence = round(sum(conf_values)/len(conf_values), 2) if len(conf_values) != 0 else 0
            # 格式化置信度数值（保留2位小数 + 分组换行）
            formatted_conf = [f"{x:.4f}" for x in conf_values]
            chunks_conf = [formatted_conf[i:i+max_per_line] 
                     for i in range(0, len(formatted_conf), max_per_line)]
            conf_lines = ["，".join(chunk) for chunk in chunks_conf]
            conf_display = "置信度列表：\n" + "\n".join(conf_lines)

            # 处理Tokens显示优化
            tokens = list(map(str, lines[2].split("：", 1)[1].split(',')))[3:]
            formatted_tokens = [f"{x}" for x in tokens]
            chunks_tokens = [formatted_tokens[i:i+max_per_line] 
                     for i in range(0, len(formatted_tokens), max_per_line)]
            tokens_lines = ["，".join(chunk) for chunk in chunks_tokens]
            tokens_display = "Token列表：\n" + "\n".join(tokens_lines)

            time_cost = lines[4].split("：", 1)[1]

            # 构建显示文本
            text = f"OCR结果：{inference}\t处理时间：{time_cost}\t平均置信度：{avg_confidence}\n{tokens_display}\n{conf_display}"

            with Image.open(img_path) as img:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except IOError:
                    raise RuntimeError(f"字体文件 {font_path} 不存在")

                # 计算文本高度（考虑多行情况）
                draw = ImageDraw.Draw(img)
                text_bbox = draw.multiline_textbbox((0, 0), text, font=font)
                padding = 10
                extension_height = text_bbox[3] + padding * 2

                # 创建新画布
                new_img = Image.new('RGB', (img.width, img.height + extension_height), (255, 255, 255))
                new_img.paste(img, (0, 0))
                draw = ImageDraw.Draw(new_img)

                # 绘制分隔线
                draw.line([(0, img.height), (img.width, img.height)], fill=(180, 180, 180), width=3)

                # 分层绘制文本
                y_position = img.height + padding
                for line in text.split('\n'):
                    draw.text((padding, y_position), line, fill=(0, 0, 0), font=font)
                    y_position += font_size + int(font_size * 0.3)  # 动态行间距

                # 保存结果
                output_path = os.path.join(output_dir, os.path.basename(img_path))
                new_img.save(output_path, quality=95, subsampling=0)
                print(f"处理完成：{output_path}")

        except Exception as e:
            print(f"处理失败：{img_path}，错误：{str(e)}")


if __name__ == "__main__":
    # 使用示例
    process_ocr_results(
        txt_path="/home/zhoukefan/GOT-OCR2.0/imgs/results.txt",  # OCR结果文件路径
        output_dir="/home/zhoukefan/GOT-OCR2.0/imgs/img_results",  # 输出目录
        font_size=13,
        max_per_line=10 # 每行能显示的置信度的数量
    )
