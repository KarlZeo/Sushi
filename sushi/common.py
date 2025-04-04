import os


class SushiError(Exception):
    """
    自定义异常类，用于在 Sushi 应用程序中抛出特定错误
    继承自标准的 Exception 类
    """
    pass


def get_extension(path):
    """
    获取文件路径的扩展名，并转换为小写
    
    参数:
        path (str): 文件路径
        
    返回:
        str: 小写的文件扩展名（包含点号）
    """
    return (os.path.splitext(path)[1]).lower()


def read_all_text(path):
    """
    读取文件的全部内容为文本
    
    参数:
        path (str): 要读取的文件路径
        
    返回:
        str: 文件的全部内容
    """
    with open(path) as file:
        return file.read()


def ensure_static_collection(value):
    """
    确保输入值为可迭代集合类型（集合、列表或元组）
    如果不是，则将其转换为列表
    
    参数:
        value: 任何值
        
    返回:
        集合类型的值，如果输入已经是集合则原样返回，否则转换为列表
    """
    if isinstance(value, (set, list, tuple)):
        return value
    return list(value)


def format_srt_time(seconds):
    """
    将秒数转换为 SRT 字幕时间格式 (HH:MM:SS,mmm)
    
    参数:
        seconds (float): 时间（秒）
        
    返回:
        str: 格式化的时间字符串，如 "00:01:23,456"
    """
    cs = round(seconds * 1000)
    return '{0:02d}:{1:02d}:{2:02d},{3:03d}'.format(
        int(cs // 3600000),
        int((cs // 60000) % 60),
        int((cs // 1000) % 60),
        int(cs % 1000))


def format_time(seconds):
    """
    将秒数转换为自定义时间格式 (H:MM:SS.cc)
    
    参数:
        seconds (float): 时间（秒）
        
    返回:
        str: 格式化的时间字符串，如 "0:01:23.45"
    """
    cs = round(seconds * 100)
    return '{0}:{1:02d}:{2:02d}.{3:02d}'.format(
            int(cs // 360000),
            int((cs // 6000) % 60),
            int((cs // 100) % 60),
            int(cs % 100))


def clip(value, minimum, maximum):
    """
    将值限制在指定的最小值和最大值范围内
    
    参数:
        value: 要限制范围的值
        minimum: 允许的最小值
        maximum: 允许的最大值
        
    返回:
        限制在指定范围内的值
    """
    return max(min(value, maximum), minimum)
