import argparse
import logging
import os
import sys
import time

# 使用绝对导入以支持 pyinstaller
# https://github.com/pyinstaller/pyinstaller/issues/2560
from sushi import run, VERSION
from sushi.common import SushiError

# 检测系统平台，为 Windows 设置控制台颜色支持
if sys.platform == 'win32':
    try:
        import colorama
        colorama.init()
        console_colors_supported = True
    except ImportError:
        console_colors_supported = False
else:
    console_colors_supported = True


class ColoredLogFormatter(logging.Formatter):
    """
    彩色日志格式化器，根据日志级别应用不同的颜色样式
    支持 ERROR（红色粗体）、WARNING（黄色粗体）、DEBUG（灰色）和默认颜色
    """
    bold_code = "\033[1m"
    reset_code = "\033[0m"
    grey_code = "\033[30m\033[1m"

    error_format = "{bold}ERROR: %(message)s{reset}".format(bold=bold_code, reset=reset_code)
    warn_format = "{bold}WARNING: %(message)s{reset}".format(bold=bold_code, reset=reset_code)
    debug_format = "{grey}%(message)s{reset}".format(grey=grey_code, reset=reset_code)
    default_format = "%(message)s"

    def format(self, record):
        """
        根据日志级别设置对应的格式
        
        参数:
            record (LogRecord): 日志记录对象
            
        返回:
            str: 格式化后的日志消息
        """
        if record.levelno == logging.DEBUG:
            self._fmt = self.debug_format
        elif record.levelno == logging.WARN:
            self._fmt = self.warn_format
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            self._fmt = self.error_format
        else:
            self._fmt = self.default_format

        return super(ColoredLogFormatter, self).format(record)


def create_arg_parser():
    """
    创建命令行参数解析器，定义所有可用的命令行参数
    
    返回:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(description='Sushi - Automatic Subtitle Shifter')

    parser.add_argument('--window', default=10, type=int, metavar='<size>', dest='window',
                        help='搜索窗口大小。 [%(default)s]')
    parser.add_argument('--max-window', default=30, type=int, metavar='<size>', dest='max_window',
                        help="Sushi 在尝试从错误中恢复时允许使用的最大搜索窗口大小。 [%(default)s]")
    parser.add_argument('--rewind-thresh', default=5, type=int, metavar='<events>', dest='rewind_thresh',
                        help="Sushi 需要遇到的连续错误数，以认为结果损坏并用更大的窗口重试。设置为 0 禁用此功能。 [%(default)s]")
    parser.add_argument('--no-grouping', action='store_false', dest='grouping',
                        help="在移动前不将事件分组。同时禁用错误恢复。")
    parser.add_argument('--max-kf-distance', default=2, type=float, metavar='<frames>', dest='max_kf_distance',
                        help='最大关键帧吸附距离。 [%(default)s]')
    parser.add_argument('--kf-mode', default='all', choices=['shift', 'snap', 'all'], dest='kf_mode',
                        help='基于关键帧的移动校正/吸附模式。 [%(default)s]')
    parser.add_argument('--smooth-radius', default=3, type=int, metavar='<events>', dest='smooth_radius',
                        help='平滑中值滤波器的半径。 [%(default)s]')

    # 23.976 帧率下的 10 帧
    parser.add_argument('--max-ts-duration', default=1001.0 / 24000.0 * 10, type=float, metavar='<seconds>',
                        dest='max_ts_duration',
                        help='被视为排版的字幕行的最大持续时间。 [%(default).3f]')
    # 23.976 帧率下的 10 帧
    parser.add_argument('--max-ts-distance', default=1001.0 / 24000.0 * 10, type=float, metavar='<seconds>',
                        dest='max_ts_distance',
                        help='要合并的两个相邻排版行之间的最大距离。 [%(default).3f]')

    # 已弃用/测试选项，请勿使用
    parser.add_argument('--test-shift-plot', default=None, dest='plot_path', help=argparse.SUPPRESS)
    parser.add_argument('--sample-type', default='uint8', choices=['float32', 'uint8'], dest='sample_type',
                        help=argparse.SUPPRESS)

    parser.add_argument('--sample-rate', default=12000, type=int, metavar='<rate>', dest='sample_rate',
                        help='降采样的音频采样率。 [%(default)s]')

    parser.add_argument('--src-audio', default=None, type=int, metavar='<id>', dest='src_audio_idx',
                        help='源视频的音频流索引')
    parser.add_argument('--src-script', default=None, type=int, metavar='<id>', dest='src_script_idx',
                        help='源视频的字幕流索引')
    parser.add_argument('--dst-audio', default=None, type=int, metavar='<id>', dest='dst_audio_idx',
                        help='目标视频的音频流索引')
    # 文件相关
    parser.add_argument('--no-cleanup', action='store_false', dest='cleanup',
                        help="不删除解复用的媒体流")
    parser.add_argument('--temp-dir', default=None, dest='temp_dir', metavar='<string>',
                        help='指定解复用媒体流时使用的临时文件夹。')
    parser.add_argument('--chapters', default=None, dest='chapters_file', metavar='<filename>',
                        help="使用指定的 XML 或 OGM 章节文件，而不是源文件中的章节。使用 'none' 禁用章节。")
    parser.add_argument('--script', default=None, dest='script_file', metavar='<filename>',
                        help='使用指定的字幕文件，而不是源文件中的字幕')

    parser.add_argument('--dst-keyframes', default=None, dest='dst_keyframes', metavar='<filename>',
                        help='目标关键帧文件')
    parser.add_argument('--src-keyframes', default=None, dest='src_keyframes', metavar='<filename>',
                        help='源关键帧文件')
    parser.add_argument('--dst-fps', default=None, type=float, dest='dst_fps', metavar='<fps>',
                        help='目标视频的帧率。如果使用关键帧，则必须提供此参数。')
    parser.add_argument('--src-fps', default=None, type=float, dest='src_fps', metavar='<fps>',
                        help='源视频的帧率。如果使用关键帧，则必须提供此参数。')
    parser.add_argument('--dst-timecodes', default=None, dest='dst_timecodes', metavar='<filename>',
                        help='使用指定的时间码文件，而不是从目标视频生成（如果可能）')
    parser.add_argument('--src-timecodes', default=None, dest='src_timecodes', metavar='<filename>',
                        help='使用指定的时间码文件，而不是从源视频生成（如果可能）')

    parser.add_argument('--src', required=True, dest="source", metavar='<filename>',
                        help='源音频/视频文件')
    parser.add_argument('--dst', required=True, dest="destination", metavar='<filename>',
                        help='目标音频/视频文件')
    parser.add_argument('-o', '--output', default=None, dest='output_script', metavar='<filename>',
                        help='输出字幕文件')

    parser.add_argument('-v', '--verbose', default=False, dest='verbose', action='store_true',
                        help='启用详细日志记录')
    parser.add_argument('--version', action='version', version=VERSION)

    return parser


def parse_args_and_run(cmd_keys):
    """
    解析命令行参数并运行 Sushi
    
    参数:
        cmd_keys (list): 命令行参数列表
    """
    def format_arg(arg):
        """格式化参数，如果包含空格则添加引号"""
        return arg if ' ' not in arg else '"{0}"'.format(arg)

    args = create_arg_parser().parse_args(cmd_keys)
    handler = logging.StreamHandler()
    if console_colors_supported and os.isatty(sys.stderr.fileno()):
        # 启用彩色输出
        handler.setFormatter(ColoredLogFormatter())
    else:
        handler.setFormatter(logging.Formatter(fmt=ColoredLogFormatter.default_format))
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    logging.info("Sushi 使用以下参数运行: {0}".format(' '.join(map(format_arg, cmd_keys))))
    start_time = time.time()
    run(args)
    logging.info('在 {0}s 内完成'.format(time.time() - start_time))


def main():
    """
    主函数：解析命令行参数并处理异常
    """
    try:
        parse_args_and_run(sys.argv[1:])
    except SushiError as e:
        logging.critical(e.args[0])
        sys.exit(2)


if __name__ == '__main__':
    main()
