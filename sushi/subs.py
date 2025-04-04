import codecs
import os
import re
import collections

from .common import SushiError, format_time, format_srt_time


def _parse_ass_time(string):
    """
    解析 ASS 格式的时间字符串为秒数
    
    参数:
        string (str): 格式为"小时:分钟:秒数"的时间字符串
        
    返回:
        float: 转换后的秒数
    """
    hours, minutes, seconds = map(float, string.split(':'))
    return hours * 3600 + minutes * 60 + seconds


class ScriptEventBase(object):
    """
    字幕事件的基类，表示单个字幕行
    提供时间调整和事件链接的功能
    """
    def __init__(self, source_index, start, end, text):
        """
        初始化字幕事件
        
        参数:
            source_index (int): 事件在原始文件中的索引
            start (float): 开始时间（秒）
            end (float): 结束时间（秒）
            text (str): 字幕文本
        """
        self.source_index = source_index
        self.start = start
        self.end = end
        self.text = text

        self._shift = 0         # 时间偏移量
        self._diff = 1          # 音频差异系数
        self._linked_event = None  # 链接到的其他事件
        self._start_shift = 0   # 额外的开始时间偏移
        self._end_shift = 0     # 额外的结束时间偏移

    @property
    def shift(self):
        """获取当前事件的时间偏移量，如果是链接事件则获取链接源的偏移量"""
        return self._linked_event.shift if self.linked else self._shift

    @property
    def diff(self):
        """获取当前事件的音频差异系数，如果是链接事件则获取链接源的差异系数"""
        return self._linked_event.diff if self.linked else self._diff

    @property
    def duration(self):
        """计算字幕持续时间（秒）"""
        return self.end - self.start

    @property
    def shifted_end(self):
        """获取应用偏移后的结束时间"""
        return self.end + self.shift + self._end_shift

    @property
    def shifted_start(self):
        """获取应用偏移后的开始时间"""
        return self.start + self.shift + self._start_shift

    def apply_shift(self):
        """将计算出的时间偏移应用到实际的开始和结束时间"""
        self.start = self.shifted_start
        self.end = self.shifted_end

    def set_shift(self, shift, audio_diff):
        """
        设置时间偏移和音频差异系数
        
        参数:
            shift (float): 时间偏移量（秒）
            audio_diff (float): 音频差异系数
        """
        assert not self.linked, '不能设置链接事件的偏移'
        self._shift = shift
        self._diff = audio_diff

    def adjust_additional_shifts(self, start_shift, end_shift):
        """
        调整额外的开始和结束时间偏移
        
        参数:
            start_shift (float): 额外的开始时间偏移
            end_shift (float): 额外的结束时间偏移
        """
        assert not self.linked, '不能为链接事件应用额外偏移'
        self._start_shift += start_shift
        self._end_shift += end_shift

    def get_link_chain_end(self):
        """
        获取链接链的末端事件
        
        返回:
            ScriptEventBase: 链接链末端的事件或自身（如果没有链接）
        """
        return self._linked_event.get_link_chain_end() if self.linked else self

    def link_event(self, other):
        """
        将当前事件链接到另一个事件
        
        参数:
            other (ScriptEventBase): 要链接到的事件
        """
        assert other.get_link_chain_end() is not self, '检测到循环链接'
        self._linked_event = other

    def resolve_link(self):
        """
        解析链接，将链接源的偏移和差异系数复制到当前事件，并移除链接
        """
        assert self.linked, '不能解析未链接的事件'
        self._shift = self._linked_event.shift
        self._diff = self._linked_event.diff
        self._linked_event = None

    @property
    def linked(self):
        """检查事件是否已链接到另一个事件"""
        return self._linked_event is not None

    def adjust_shift(self, value):
        """
        调整偏移量
        
        参数:
            value (float): 要添加的偏移量
        """
        assert not self.linked, '不能调整链接事件的时间'
        self._shift += value


class ScriptBase(object):
    """
    字幕脚本的基类，提供事件列表的基本操作
    """
    def __init__(self, events):
        """
        初始化脚本对象
        
        参数:
            events (list): 字幕事件列表
        """
        self.events = events

    def sort_by_time(self):
        """按开始时间对事件列表进行排序"""
        self.events.sort(key=lambda x: x.start)


class SrtEvent(ScriptEventBase):
    """
    SRT 格式字幕事件，继承自 ScriptEventBase
    """
    is_comment = False  # SRT 没有注释功能
    style = None        # SRT 没有样式功能

    # 用于从文本中提取 SRT 事件的正则表达式
    EVENT_REGEX = re.compile(r"""
                               (\d+?)\s+? # line-number
                               (\d{1,2}:\d{1,2}:\d{1,2},\d+)\s-->\s(\d{1,2}:\d{1,2}:\d{1,2},\d+).  # timestamp
                               (.+?) # actual text
                           (?= # lookahead for the next line or end of the file
                               (?:\d+?\s+? # line-number
                               \d{1,2}:\d{1,2}:\d{1,2},\d+\s-->\s\d{1,2}:\d{1,2}:\d{1,2},\d+) # timestamp
                               |$
                           )""", flags=re.VERBOSE | re.DOTALL)

    @classmethod
    def from_string(cls, text):
        """
        从字符串创建 SRT 事件
        
        参数:
            text (str): 包含 SRT 事件的文本
            
        返回:
            SrtEvent: 新创建的 SRT 事件
        """
        match = cls.EVENT_REGEX.match(text)
        start = cls.parse_time(match.group(2))
        end = cls.parse_time(match.group(3))
        return SrtEvent(int(match.group(1)), start, end, match.group(4).strip())

    def __str__(self):
        """
        将 SRT 事件转换为字符串格式
        
        返回:
            str: SRT 格式的事件字符串
        """
        return '{0}\n{1} --> {2}\n{3}'.format(self.source_index, self._format_time(self.start),
                                              self._format_time(self.end), self.text)

    @staticmethod
    def parse_time(time_string):
        """
        解析 SRT 格式的时间字符串
        
        参数:
            time_string (str): SRT 格式的时间字符串，如 "01:23:45,678"
            
        返回:
            float: 转换后的秒数
        """
        return _parse_ass_time(time_string.replace(',', '.'))

    @staticmethod
    def _format_time(seconds):
        """
        将秒数格式化为 SRT 时间格式
        
        参数:
            seconds (float): 秒数
            
        返回:
            str: SRT 格式的时间字符串，如 "01:23:45,678"
        """
        return format_srt_time(seconds)


class SrtScript(ScriptBase):
    """
    SRT 格式字幕脚本，继承自 ScriptBase
    """
    @classmethod
    def from_file(cls, path):
        """
        从文件加载 SRT 脚本
        
        参数:
            path (str): SRT 文件的路径
            
        返回:
            SrtScript: 包含所有事件的脚本对象
            
        异常:
            SushiError: 当文件不存在时抛出
        """
        try:
            with codecs.open(path, encoding='utf-8-sig') as script:
                text = script.read()
                events_list = [SrtEvent(
                    source_index=int(match.group(1)),
                    start=SrtEvent.parse_time(match.group(2)),
                    end=SrtEvent.parse_time(match.group(3)),
                    text=match.group(4).strip()
                ) for match in SrtEvent.EVENT_REGEX.finditer(text)]
                return cls(events_list)
        except IOError:
            raise SushiError("Script {0} not found".format(path))

    def save_to_file(self, path):
        """
        将脚本保存到文件
        
        参数:
            path (str): 要保存到的文件路径
        """
        text = '\n\n'.join(map(str, self.events))
        with codecs.open(path, encoding='utf-8', mode='w') as script:
            script.write(text)


class AssEvent(ScriptEventBase):
    """
    ASS 格式字幕事件，继承自 ScriptEventBase
    添加了 ASS 格式特有的属性
    """
    def __init__(self, text, position=0):
        """
        初始化 ASS 事件
        
        参数:
            text (str): ASS 格式的事件文本行
            position (int): 事件在原始文件中的位置
        """
        kind, _, rest = text.partition(':')
        split = [x.strip() for x in rest.split(',', 9)]

        super(AssEvent, self).__init__(
            source_index=position,
            start=_parse_ass_time(split[1]),
            end=_parse_ass_time(split[2]),
            text=split[9]
        )
        self.kind = kind
        self.is_comment = self.kind.lower() == 'comment'
        self.layer = split[0]
        self.style = split[3]
        self.name = split[4]
        self.margin_left = split[5]
        self.margin_right = split[6]
        self.margin_vertical = split[7]
        self.effect = split[8]

    def __str__(self):
        """
        将 ASS 事件转换为字符串格式
        
        返回:
            str: ASS 格式的事件字符串
        """
        return '{0}: {1},{2},{3},{4},{5},{6},{7},{8},{9},{10}'.format(self.kind, self.layer,
                                                                      self._format_time(self.start),
                                                                      self._format_time(self.end),
                                                                      self.style, self.name,
                                                                      self.margin_left, self.margin_right,
                                                                      self.margin_vertical, self.effect,
                                                                      self.text)

    @staticmethod
    def _format_time(seconds):
        """
        将秒数格式化为 ASS 时间格式
        
        参数:
            seconds (float): 秒数
            
        返回:
            str: ASS 格式的时间字符串，如 "1:23:45.67"
        """
        return format_time(seconds)


class AssScript(ScriptBase):
    """
    ASS 格式字幕脚本，继承自 ScriptBase
    包含脚本信息、样式、事件和其他自定义部分
    """
    def __init__(self, script_info, styles, events, other):
        """
        初始化 ASS 脚本
        
        参数:
            script_info (list): 脚本信息部分的行列表
            styles (list): 样式部分的行列表
            events (list): 事件对象列表
            other (collections.OrderedDict): 其他自定义部分的有序字典
        """
        super(AssScript, self).__init__(events)
        self.script_info = script_info
        self.styles = styles
        self.other = other

    @classmethod
    def from_file(cls, path):
        """
        从文件加载 ASS 脚本
        
        参数:
            path (str): ASS 文件的路径
            
        返回:
            AssScript: 包含所有部分的脚本对象
            
        异常:
            SushiError: 当文件不存在或格式不正确时抛出
        """
        script_info, styles, events = [], [], []
        other_sections = collections.OrderedDict()

        def parse_script_info_line(line):
            """解析脚本信息部分的行"""
            if line.startswith('Format:'):
                return
            script_info.append(line)

        def parse_styles_line(line):
            """解析样式部分的行"""
            if line.startswith('Format:'):
                return
            styles.append(line)

        def parse_event_line(line):
            """解析事件部分的行"""
            if line.startswith('Format:'):
                return
            events.append(AssEvent(line, position=len(events) + 1))

        def create_generic_parse(section_name):
            """创建处理自定义部分的函数"""
            if section_name in other_sections:
                raise SushiError("检测到重复的部分，无效的脚本？")
            other_sections[section_name] = []
            return other_sections[section_name].append

        parse_function = None

        try:
            with codecs.open(path, encoding='utf-8-sig') as script:
                for line_idx, line in enumerate(script):
                    line = line.strip()
                    if not line:
                        continue
                    low = line.lower()
                    if low == '[script info]':
                        parse_function = parse_script_info_line
                    elif low == '[v4+ styles]':
                        parse_function = parse_styles_line
                    elif low == '[events]':
                        parse_function = parse_event_line
                    elif re.match(r'\[.+?\]', low):
                        parse_function = create_generic_parse(line)
                    elif not parse_function:
                        raise SushiError("这是无效的 ASS 脚本")
                    else:
                        try:
                            parse_function(line)
                        except Exception as e:
                            raise SushiError("这是无效的 ASS 脚本: {0} [第 {1} 行]".format(e.message, line_idx))
        except IOError:
            raise SushiError("找不到脚本 {0}".format(path))
        return cls(script_info, styles, events, other_sections)

    def save_to_file(self, path):
        """
        将 ASS 脚本保存到文件
        
        参数:
            path (str): 要保存到的文件路径
        """
        # if os.path.exists(path):
        #     raise RuntimeError('File %s already exists' % path)
        lines = []
        if self.script_info:
            lines.append('[Script Info]')
            lines.extend(self.script_info)
            lines.append('')

        if self.styles:
            lines.append('[V4+ Styles]')
            lines.append('Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding')
            lines.extend(self.styles)
            lines.append('')

        if self.events:
            events = sorted(self.events, key=lambda x: x.source_index)
            lines.append('[Events]')
            lines.append('Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text')
            lines.extend(map(str, events))

        if self.other:
            for section_name, section_lines in self.other.items():
                lines.append('')
                lines.append(section_name)
                lines.extend(section_lines)

        with codecs.open(path, encoding='utf-8-sig', mode='w') as script:
            script.write(str(os.linesep).join(lines))
