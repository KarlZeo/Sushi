from .common import SushiError, read_all_text


def parse_scxvid_keyframes(text):
    """
    从 XviD 格式的关键帧文件内容中提取关键帧位置
    
    参数:
        text (str): XviD 统计文件的内容
        
    返回:
        list: 关键帧位置列表（帧索引减去3，因为XviD文件有3行头部信息）
    """
    return [i - 3 for i, line in enumerate(text.splitlines()) if line and line[0] == 'i']


def parse_keyframes(path):
    """
    解析关键帧文件并提取关键帧位置
    
    参数:
        path (str): 关键帧文件路径
        
    返回:
        list: 关键帧位置列表
        
    异常:
        SushiError: 当关键帧文件格式不支持时抛出
    """
    text = read_all_text(path)
    if '# XviD 2pass stat file' in text:
        frames = parse_scxvid_keyframes(text)
    else:
        raise SushiError('Unsupported keyframes type')
    if 0 not in frames:
        frames.insert(0, 0)
    return frames
