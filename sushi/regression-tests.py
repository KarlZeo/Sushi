from contextlib import contextmanager
import json
import logging
import os
import gc
import sys
import resource
import re
import subprocess
import argparse

from .common import format_time
from .demux import Timecodes
from .subs import AssScript
from .wav import WavStream


root_logger = logging.getLogger('')  # 获取根日志记录器


def strip_tags(text):
    """
    移除字幕文本中的标签（花括号中的内容）
    
    参数:
        text (str): 包含标签的字幕文本
        
    返回:
        str: 移除标签后的文本
    """
    return re.sub(r'{.*?}', " ", text)


@contextmanager
def set_file_logger(path):
    """
    设置文件日志记录器的上下文管理器
    
    参数:
        path (str): 日志文件路径
        
    用法:
        with set_file_logger('path/to/log.txt'):
            # 执行需要记录日志的操作
    """
    handler = logging.FileHandler(path, mode='a')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(handler)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)


def compare_scripts(ideal_path, test_path, timecodes, test_name, expected_errors):
    """
    比较理想字幕文件和测试字幕文件的差异
    
    参数:
        ideal_path (str): 理想字幕文件的路径
        test_path (str): 测试字幕文件的路径
        timecodes (Timecodes): 时间码对象，用于帧号和时间的转换
        test_name (str): 测试名称，用于日志记录
        expected_errors (int): 预期的错误数量
        
    返回:
        bool: 如果错误数量等于预期错误数量，则返回 True，否则返回 False
    """
    ideal_script = AssScript.from_file(ideal_path)
    test_script = AssScript.from_file(test_path)
    if len(test_script.events) != len(ideal_script.events):
        logging.critical("Script length didn't match: {0} in ideal vs {1} in test. Test {2}".format(
            len(ideal_script.events), len(test_script.events), test_name)
        )
        return False
    ideal_script.sort_by_time()
    test_script.sort_by_time()
    failed = 0
    ft = format_time
    for idx, (ideal, test) in enumerate(zip(ideal_script.events, test_script.events)):
        ideal_start_frame = timecodes.get_frame_number(ideal.start)
        ideal_end_frame = timecodes.get_frame_number(ideal.end)

        test_start_frame = timecodes.get_frame_number(test.start)
        test_end_frame = timecodes.get_frame_number(test.end)

        if ideal_start_frame != test_start_frame and ideal_end_frame != test_end_frame:
            logging.debug('{0}: start and end time failed at "{1}". {2}-{3} vs {4}-{5}'.format(
                idx, strip_tags(ideal.text), ft(ideal.start), ft(ideal.end), ft(test.start), ft(test.end))
            )
            failed += 1
        elif ideal_end_frame != test_end_frame:
            logging.debug(
                '{0}: end time failed at "{1}". {2} vs {3}'.format(
                    idx, strip_tags(ideal.text), ft(ideal.end), ft(test.end))
            )
            failed += 1
        elif ideal_start_frame != test_start_frame:
            logging.debug(
                '{0}: start time failed at "{1}". {2} vs {3}'.format(
                    idx, strip_tags(ideal.text), ft(ideal.start), ft(test.start))
            )
            failed += 1

    logging.info('Total lines: {0}, good: {1}, failed: {2}'.format(len(ideal_script.events), len(ideal_script.events) - failed, failed))

    if failed > expected_errors:
        logging.critical('Got more failed lines than expected ({0} actual vs {1} expected)'.format(failed, expected_errors))
        return False
    elif failed < expected_errors:
        logging.critical('Got less failed lines than expected ({0} actual vs {1} expected)'.format(failed, expected_errors))
        return False
    else:
        logging.critical('Met expectations')
        return True


def run_test(base_path, plots_path, test_name, params):
    """
    运行单个测试
    
    参数:
        base_path (str): 测试基础路径
        plots_path (str): 图表输出路径
        test_name (str): 测试名称
        params (dict): 测试参数字典
        
    返回:
        bool: 测试是否成功
    """
    def safe_add_key(args, key, name):
        """
        安全地向命令行参数列表中添加键值对
        """
        if name in params:
            args.extend((key, str(params[name])))

    def safe_add_path(args, folder, key, name):
        """
        安全地向命令行参数列表中添加路径参数
        """
        if name in params:
            args.extend((key, os.path.join(folder, params[name])))

    logging.info('Testing "{0}"'.format(test_name))

    folder = os.path.join(base_path, params['folder'])

    cmd = ["sushi"]

    # 添加各种命令行参数
    safe_add_path(cmd, folder, '--src', 'src')
    safe_add_path(cmd, folder, '--dst', 'dst')
    safe_add_path(cmd, folder, '--src-keyframes', 'src-keyframes')
    safe_add_path(cmd, folder, '--dst-keyframes', 'dst-keyframes')
    safe_add_path(cmd, folder, '--src-timecodes', 'src-timecodes')
    safe_add_path(cmd, folder, '--dst-timecodes', 'dst-timecodes')
    safe_add_path(cmd, folder, '--script', 'script')
    safe_add_path(cmd, folder, '--chapters', 'chapters')
    safe_add_path(cmd, folder, '--src-script', 'src-script')
    safe_add_path(cmd, folder, '--dst-script', 'dst-script')
    safe_add_key(cmd, '--max-kf-distance', 'max-kf-distance')
    safe_add_key(cmd, '--max-ts-distance', 'max-ts-distance')
    safe_add_key(cmd, '--max-ts-duration', 'max-ts-duration')

    # 设置输出路径和测试图表路径
    output_path = os.path.join(folder, params['dst']) + '.sushi.test.ass'
    cmd.extend(('-o', output_path))
    if plots_path:
        cmd.extend(('--test-shift-plot', os.path.join(plots_path, '{0}.png'.format(test_name))))

    log_path = os.path.join(folder, 'sushi_test.log')

    # 执行 sushi 命令并捕获输出到日志文件
    with open(log_path, "w") as log_file:
        try:
            subprocess.call(cmd, stderr=log_file, stdout=log_file)
        except Exception as e:
            logging.critical('Sushi failed on test "{0}": {1}'.format(test_name, e.message))
            return False

    # 比较输出脚本与理想脚本
    with set_file_logger(log_path):
        ideal_path = os.path.join(folder, params['ideal'])
        try:
            timecodes = Timecodes.from_file(os.path.join(folder, params['dst-timecodes']))
        except KeyError:
            timecodes = Timecodes.cfr(params['fps'])

        return compare_scripts(ideal_path, output_path, timecodes, test_name, params['expected_errors'])


def run_wav_test(test_name, file_path, params):
    """
    运行 WAV 文件加载性能测试
    
    参数:
        test_name (str): 测试名称
        file_path (str): WAV 文件路径
        params (dict): 测试参数字典
        
    返回:
        bool: 测试是否成功（是否满足性能要求）
    """
    gc.collect(2)  # 强制进行垃圾收集

    # 记录加载 WAV 前后的资源使用情况
    before = resource.getrusage(resource.RUSAGE_SELF)
    _ = WavStream(file_path, params.get('sample_rate', 12000), params.get('sample_type', 'uint8'))
    after = resource.getrusage(resource.RUSAGE_SELF)

    # 计算消耗的时间和内存
    total_time = (after.ru_stime - before.ru_stime) + (after.ru_utime - before.ru_utime)
    ram_difference = (after.ru_maxrss - before.ru_maxrss) / 1024.0 / 1024.0

    # 检查是否满足性能要求
    if 'max_time' in params and total_time > params['max_time']:
        logging.critical('Loading "{0}" took too much time: {1} vs {2} seconds'
                         .format(test_name, total_time, params['max_time']))
        return False
    if 'max_memory' in params and ram_difference > params['max_memory']:
        logging.critical('Loading "{0}" consumed too much RAM: {1} vs {2}'
                         .format(test_name, ram_difference, params['max_memory']))
        return False
    return True


def create_arg_parser():
    """
    创建命令行参数解析器
    
    返回:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(description='Sushi regression testing util')

    parser.add_argument('--only', dest="run_only", nargs="*", metavar='<test names>',
                        help='Test names to run')
    parser.add_argument('-c', '--conf', default="tests.json", dest='conf_path', metavar='<filename>',
                        help='Config file path')

    return parser


def run():
    """
    主函数：运行所有回归测试
    """
    # 设置日志记录
    root_logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(console_handler)

    # 解析命令行参数
    args = create_arg_parser().parse_args()

    # 加载配置文件
    try:
        with open(args.conf_path) as file:
            config = json.load(file)
    except IOError as e:
        logging.critical(e)
        sys.exit(2)

    def should_run(name):
        """判断是否应该运行指定名称的测试"""
        return not args.run_only or name in args.run_only

    # 运行测试并统计结果
    failed = ran = 0
    for test_name, params in config.get('tests', {}).items():
        if not should_run(test_name):
            continue
        if not params.get('disabled', False):
            ran += 1
            if not run_test(config['basepath'], config['plots'], test_name, params):
                failed += 1
            logging.info('')
        else:
            logging.warn('Test "{0}" disabled'.format(test_name))

    # 运行 WAV 文件测试
    if should_run("wavs"):
        for test_name, params in config.get('wavs', {}).items():
            ran += 1
            if not run_wav_test(test_name, os.path.join(config['basepath'], params['file']), params):
                failed += 1
            logging.info('')

    # 输出测试结果摘要
    logging.info('Ran {0} tests, {1} failed'.format(ran, failed))


if __name__ == '__main__':
    run()
