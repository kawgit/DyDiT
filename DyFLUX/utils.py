from PIL import Image
import os

def load_txt_to_list(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # 去除每行末尾的换行符
            result = [line.strip() for line in lines]
            return result
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}。")
    except Exception as e:
        print(f"发生未知错误：{e}")
    return []

def append_line_to_file(file_path, line_to_append):
    """
    在指定路径的文件中追加一行文本。
    
    :param file_path: 文件的完整路径
    :param line_to_append: 要追加的文本行
    """
    try:
        # 检查文件所在目录是否存在，如果不存在则创建
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # 打开文件并追加一行文本
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(line_to_append + '\n')
        # print(f"成功写入：'{line_to_append}' 到文件 '{file_path}'")
    
    except Exception as e:
        print(f"写入文件时发生错误：{e}")


def combine_images_horizontally(image_paths):
    """将给定路径中的图片横向拼接"""
    images = [Image.open(path) for path in image_paths]
    widths, heights = zip(*(i.size for i in images))
    
    # 计算新图像的大小
    total_width = sum(widths)
    max_height = max(heights)
    
    # 创建空白图像
    combined_image = Image.new('RGB', (total_width, max_height))
    
    # 拼接图片
    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return combined_image

