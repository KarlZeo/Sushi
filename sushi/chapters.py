import re
from . import common


def parse_times(times):
    """
    解析时间字符串列表并转换为秒数
    
    参数:
        times (list): 格式为 "小时:分钟:秒数" 的时间字符串列表
        
    返回:
        list: 按升序排序的章节开始时间（秒数）列表，如果列表中没有 0 秒的章节，会自动添加
    """
    result = []
    for t in times:
        hours, minutes, seconds = list(map(float, t.split(':')))
        result.append(hours * 3600 + minutes * 60 + seconds)

    result.sort()
    if result[0] != 0:
        result.insert(0, 0)
    return result


def parse_xml_start_times(text):
    """
    从 XML 格式的章节文件内容中提取章节开始时间
    
    参数:
        text (str): XML 格式的章节文件内容
        
    返回:
        list: 章节开始时间（秒数）列表
    """
    times = re.findall(r'<ChapterTimeStart>(\d+:\d+:\d+\.\d+)</ChapterTimeStart>', text)
    return parse_times(times)


def get_xml_start_times(path):
    """
    从 XML 格式的章节文件中获取章节开始时间
    
    参数:
        path (str): XML 文件路径
        
    返回:
        list: 章节开始时间（秒数）列表
    """
    return parse_xml_start_times(common.read_all_text(path))


def parse_ogm_start_times(text):
    """
    从 OGM 格式的章节文件内容中提取章节开始时间
    
    参数:
        text (str): OGM 格式的章节文件内容
        
    返回:
        list: 章节开始时间（秒数）列表
    """
    times = re.findall(r'CHAPTER\d+=(\d+:\d+:\d+\.\d+)', text, flags=re.IGNORECASE)
    return parse_times(times)


def get_ogm_start_times(path):
    """
    从 OGM 格式的章节文件中获取章节开始时间
    
    参数:
        path (str): OGM 文件路径
        
    返回:
        list: 章节开始时间（秒数）列表
    """
    return parse_ogm_start_times(common.read_all_text(path))


def format_ogm_chapters(start_times):
    """
    将章节开始时间列表格式化为 OGM 格式的章节字符串
    
    参数:
        start_times (list): 章节开始时间（秒数）列表
        
    返回:
        str: OGM 格式的章节字符串，包含编号的章节时间和空章节名
    """
    return "\n".join("CHAPTER{0:02}={1}\nCHAPTER{0:02}NAME=".format(idx+1, common.format_srt_time(start).replace(',', '.'))
                     for idx, start in enumerate(start_times)) + "\n"
