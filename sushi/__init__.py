import bisect
from itertools import takewhile, chain
import logging
import operator
import os

import numpy as np

from . import chapters
from .common import SushiError, get_extension, format_time, ensure_static_collection
from .demux import Timecodes, Demuxer
from . import keyframes
from .subs import AssScript, SrtScript
from .wav import WavStream


try:
    import matplotlib.pyplot as plt
    plot_enabled = True
except ImportError:
    plot_enabled = False


ALLOWED_ERROR = 0.01  # 允许的误差范围（秒）
MAX_GROUP_STD = 0.025  # 组内偏移的最大标准差
VERSION = '0.6.2'  # 软件版本号


def abs_diff(a, b):
    """
    计算两个数的绝对差值
    
    参数:
        a, b: 要比较的两个数值
        
    返回:
        两个数的绝对差值
    """
    return abs(a - b)


def interpolate_nones(data, points):
    """
    对数据中的 None 值进行插值
    
    参数:
        data: 包含可能有 None 值的数据列表
        points: 与数据对应的坐标点
        
    返回:
        插值后的数据列表，所有 None 值被替换为插值结果
    """
    data = ensure_static_collection(data)
    values_lookup = {p: v for p, v in zip(points, data) if v is not None}
    if not values_lookup:
        return []

    zero_points = {p for p, v in zip(points, data) if v is None}
    if not zero_points:
        return data

    data_list = sorted(values_lookup.items())
    zero_points = sorted(x for x in zero_points if x not in values_lookup)

    out = np.interp(x=zero_points,
                    xp=list(map(operator.itemgetter(0), data_list)),
                    fp=list(map(operator.itemgetter(1), data_list)))

    values_lookup.update(zip(zero_points, out))

    return [
        values_lookup[point] if value is None else value
        for point, value in zip(points, data)
    ]


def running_median(values, window_size):
    """
    计算滑动窗口中值滤波
    
    参数:
        values: 输入值列表
        window_size: 中值滤波窗口大小，必须为奇数
        
    返回:
        应用中值滤波后的列表
    
    异常:
        SushiError: 当窗口大小不是奇数时抛出
    """
    if window_size % 2 != 1:
        raise SushiError('Median window size should be odd')
    half_window = window_size // 2
    medians = []
    items_count = len(values)
    for idx in range(items_count):
        radius = min(half_window, idx, items_count - idx - 1)
        med = np.median(values[idx - radius:idx + radius + 1])
        medians.append(med)
    return medians


def smooth_events(events, radius):
    """
    对事件的时间偏移进行平滑处理
    
    参数:
        events: 要平滑的事件列表
        radius: 平滑窗口的半径，窗口大小为 2*radius+1
    """
    if not radius:
        return
    window_size = radius * 2 + 1
    shifts = [e.shift for e in events]
    smoothed = running_median(shifts, window_size)
    for event, new_shift in zip(events, smoothed):
        event.set_shift(new_shift, event.diff)


def detect_groups(events_iter):
    """
    根据时间偏移将事件划分为组
    
    参数:
        events_iter: 事件的迭代器
        
    返回:
        list: 组列表，每个组是事件的列表
    """
    events_iter = iter(events_iter)
    groups_list = [[next(events_iter)]]
    for event in events_iter:
        if abs_diff(event.shift, groups_list[-1][-1].shift) > ALLOWED_ERROR:
            groups_list.append([])
        groups_list[-1].append(event)
    return groups_list


def groups_from_chapters(events, times):
    """
    根据章节时间点将事件划分为组
    
    参数:
        events: 事件列表
        times: 章节开始时间的列表
        
    返回:
        list: 按章节划分的事件组列表
    """
    logging.info('Chapter start points: {0}'.format([format_time(t) for t in times]))
    groups = [[]]
    chapter_times = iter(times[1:] + [36000000000])  # 添加一个非常大的结束时间
    current_chapter = next(chapter_times)

    for event in events:
        if event.end > current_chapter:
            groups.append([])
            while event.end > current_chapter:
                current_chapter = next(chapter_times)

        groups[-1].append(event)

    groups = [g for g in groups if g]  # 过滤掉空组
    # 检查是否有只包含链接事件的组（例如只有注释的章节）
    broken_groups = [group for group in groups if not any(e for e in group if not e.linked)]
    if broken_groups:
        for group in broken_groups:
            for event in group:
                parent = event.get_link_chain_end()
                parent_group = next(group for group in groups if parent in group)
                parent_group.append(event)
            del group[:]
        groups = [g for g in groups if g]
        # 重新排序，因为添加链接事件可能打乱顺序
        for group in groups:
            group.sort(key=lambda event: event.start)

    return groups


def split_broken_groups(groups):
    """
    分割可能被错误分组的事件组
    
    参数:
        groups: 事件组列表
        
    返回:
        list: 修正后的事件组列表
    """
    correct_groups = []
    broken_found = False
    for g in groups:
        std = np.std([e.shift for e in g])
        if std > MAX_GROUP_STD:
            logging.warn('Shift is not consistent between {0} and {1}, most likely chapters are wrong (std: {2}). '
                         'Switching to automatic grouping.'.format(format_time(g[0].start), format_time(g[-1].end),
                                                                   std))
            correct_groups.extend(detect_groups(g))
            broken_found = True
        else:
            correct_groups.append(g)

    if broken_found:
        groups_iter = iter(correct_groups)
        correct_groups = [list(next(groups_iter))]
        for group in groups_iter:
            if abs_diff(correct_groups[-1][-1].shift, group[0].shift) >= ALLOWED_ERROR \
                    or np.std([e.shift for e in group + correct_groups[-1]]) >= MAX_GROUP_STD:
                correct_groups.append([])

            correct_groups[-1].extend(group)
    return correct_groups


def fix_near_borders(events):
    """
    修复靠近边界的事件，假设所有差异大于 5 倍中位差的行都是有问题的
    
    参数:
        events: 事件列表
    """
    def fix_border(event_list, median_diff):
        last_ten_diff = np.median([x.diff for x in event_list[:10]], overwrite_input=True)
        diff_limit = min(last_ten_diff, median_diff)
        broken = []
        for event in event_list:
            if not 0.2 < (event.diff / diff_limit) < 5:
                broken.append(event)
            else:
                for x in broken:
                    x.link_event(event)
                return len(broken)
        return 0

    median_diff = np.median([x.diff for x in events], overwrite_input=True)

    fixed_count = fix_border(events, median_diff)
    if fixed_count:
        logging.info('Fixing {0} border events right after {1}'.format(fixed_count, format_time(events[0].start)))

    fixed_count = fix_border(list(reversed(events)), median_diff)
    if fixed_count:
        logging.info('Fixing {0} border events right before {1}'.format(fixed_count, format_time(events[-1].end)))


def get_distance_to_closest_kf(timestamp, keyframes):
    """
    计算时间戳到最近关键帧的距离
    
    参数:
        timestamp: 时间戳（秒）
        keyframes: 关键帧时间点列表
        
    返回:
        float: 到最近关键帧的距离（秒）
    """
    idx = bisect.bisect_left(keyframes, timestamp)
    if idx == 0:
        kf = keyframes[0]
    elif idx == len(keyframes):
        kf = keyframes[-1]
    else:
        before = keyframes[idx - 1]
        after = keyframes[idx]
        kf = after if after - timestamp < timestamp - before else before
    return kf - timestamp


def find_keyframe_shift(group, src_keytimes, dst_keytimes, src_timecodes, dst_timecodes, max_kf_distance):
    """
    计算关键帧之间的偏移
    
    参数:
        group: 事件组
        src_keytimes: 源视频关键帧时间点
        dst_keytimes: 目标视频关键帧时间点
        src_timecodes: 源视频时间码
        dst_timecodes: 目标视频时间码
        max_kf_distance: 最大关键帧距离
        
    返回:
        tuple: (开始偏移, 结束偏移)
    """
    def get_distance(src_distance, dst_distance, limit):
        if abs(dst_distance) > limit:
            return None
        shift = dst_distance - src_distance
        return shift if abs(shift) < limit else None

    src_start = get_distance_to_closest_kf(group[0].start, src_keytimes)
    src_end = get_distance_to_closest_kf(group[-1].end + src_timecodes.get_frame_size(group[-1].end), src_keytimes)

    dst_start = get_distance_to_closest_kf(group[0].shifted_start, dst_keytimes)
    dst_end = get_distance_to_closest_kf(group[-1].shifted_end + dst_timecodes.get_frame_size(group[-1].end), dst_keytimes)

    snapping_limit_start = src_timecodes.get_frame_size(group[0].start) * max_kf_distance
    snapping_limit_end = src_timecodes.get_frame_size(group[0].end) * max_kf_distance

    return (get_distance(src_start, dst_start, snapping_limit_start),
            get_distance(src_end, dst_end, snapping_limit_end))


def find_keyframes_distances(event, src_keytimes, dst_keytimes, timecodes, max_kf_distance):
    """
    计算单个事件开始和结束时间与关键帧的距离
    
    参数:
        event: 要处理的事件
        src_keytimes: 源视频关键帧时间点
        dst_keytimes: 目标视频关键帧时间点
        timecodes: 时间码
        max_kf_distance: 最大关键帧距离
        
    返回:
        tuple: (开始时间偏移, 结束时间偏移)
    """
    def find_keyframe_distance(src_time, dst_time):
        src = get_distance_to_closest_kf(src_time, src_keytimes)
        dst = get_distance_to_closest_kf(dst_time, dst_keytimes)
        snapping_limit = timecodes.get_frame_size(src_time) * max_kf_distance

        if abs(src) < snapping_limit and abs(dst) < snapping_limit and abs(src - dst) < snapping_limit:
            return dst - src
        return 0

    ds = find_keyframe_distance(event.start, event.shifted_start)
    de = find_keyframe_distance(event.end, event.shifted_end)
    return ds, de


def snap_groups_to_keyframes(events, chapter_times, max_ts_duration, max_ts_distance, src_keytimes, dst_keytimes,
                             src_timecodes, dst_timecodes, max_kf_distance, kf_mode):
    """
    将事件组吸附到关键帧
    
    参数:
        events: 事件列表
        chapter_times: 章节时间点
        max_ts_duration: 最大排版持续时间
        max_ts_distance: 最大排版距离
        src_keytimes: 源视频关键帧时间点
        dst_keytimes: 目标视频关键帧时间点
        src_timecodes: 源视频时间码
        dst_timecodes: 目标视频时间码
        max_kf_distance: 最大关键帧距离
        kf_mode: 关键帧模式（'all', 'shift', 'snap'）
    """
    if not max_kf_distance:
        return

    groups = merge_short_lines_into_groups(events, chapter_times, max_ts_duration, max_ts_distance)

    if kf_mode == 'all' or kf_mode == 'shift':
        #  步骤 1: 不改变事件持续时间的情况下吸附事件。用于修正轻微的音频不精确
        shifts = []
        times = []
        for group in groups:
            shifts.extend(find_keyframe_shift(group, src_keytimes, dst_keytimes, src_timecodes, dst_timecodes, max_kf_distance))
            times.extend((group[0].shifted_start, group[-1].shifted_end))

        shifts = interpolate_nones(shifts, times)
        if shifts:
            mean_shift = np.mean(shifts)
            shifts = zip(*[iter(shifts)] * 2)

            logging.info('Group {0}-{1} corrected by {2}'.format(format_time(events[0].start), format_time(events[-1].end), mean_shift))
            for group, (start_shift, end_shift) in zip(groups, shifts):
                if abs(start_shift - end_shift) > 0.001 and len(group) > 1:
                    actual_shift = min(start_shift, end_shift, key=lambda x: abs(x - mean_shift))
                    logging.warning("Typesetting group at {0} had different shift at start/end points ({1} and {2}). Shifting by {3}."
                                    .format(format_time(group[0].start), start_shift, end_shift, actual_shift))
                    for e in group:
                        e.adjust_shift(actual_shift)
                else:
                    for e in group:
                        e.adjust_additional_shifts(start_shift, end_shift)

    if kf_mode == 'all' or kf_mode == 'snap':
        # 步骤 2: 分别吸附开始和结束时间
        for group in groups:
            if len(group) > 1:
                pass  # 不吸附排版组
            start_shift, end_shift = find_keyframes_distances(group[0], src_keytimes, dst_keytimes, src_timecodes, max_kf_distance)
            if abs(start_shift) > 0.01 or abs(end_shift) > 0.01:
                logging.info('Snapping {0} to keyframes, start time by {1}, end: {2}'.format(format_time(group[0].start), start_shift, end_shift))
                group[0].adjust_additional_shifts(start_shift, end_shift)


def average_shifts(events):
    """
    计算事件偏移的加权平均值
    
    参数:
        events: 事件列表
        
    返回:
        float: 平均偏移值
    """
    events = [e for e in events if not e.linked]
    shifts = [x.shift for x in events]
    weights = [1 - x.diff for x in events]
    if shifts != [] or weights != []:
        avg = np.average(shifts, weights=weights)
    else:
        avg = [0]
    for e in events:
        e.set_shift(avg, e.diff)
    return avg


def merge_short_lines_into_groups(events, chapter_times, max_ts_duration, max_ts_distance):
    """
    将短行合并成组（用于排版和关键帧吸附）
    
    参数:
        events: 事件列表
        chapter_times: 章节时间点
        max_ts_duration: 最大排版持续时间
        max_ts_distance: 最大排版距离
        
    返回:
        list: 分组后的事件列表
    """
    search_groups = []
    chapter_times = iter(chapter_times[1:] + [100000000])
    next_chapter = next(chapter_times)
    events = ensure_static_collection(events)

    processed = set()
    for idx, event in enumerate(events):
        if idx in processed:
            continue

        while event.end > next_chapter:
            next_chapter = next(chapter_times)

        if event.duration > max_ts_duration:
            search_groups.append([event])
            processed.add(idx)
        else:
            group = [event]
            group_end = event.end
            i = idx + 1
            while i < len(events) and abs(group_end - events[i].start) < max_ts_distance:
                if events[i].end < next_chapter and events[i].duration <= max_ts_duration:
                    processed.add(i)
                    group.append(events[i])
                    group_end = max(group_end, events[i].end)
                i += 1

            search_groups.append(group)

    return search_groups


def prepare_search_groups(events, source_duration, chapter_times, max_ts_duration, max_ts_distance):
    """
    准备用于音频搜索的组
    
    参数:
        events: 事件列表
        source_duration: 源音频持续时间
        chapter_times: 章节时间点
        max_ts_duration: 最大排版持续时间
        max_ts_distance: 最大排版距离
        
    返回:
        list: 分组后的事件列表
    """
    last_unlinked = None
    for idx, event in enumerate(events):
        if event.is_comment:
            try:
                event.link_event(events[idx + 1])
            except IndexError:
                event.link_event(last_unlinked)
            continue
        if (event.start + event.duration / 2.0) > source_duration:
            logging.info('Event time outside of audio range, ignoring: %s', event)
            event.link_event(last_unlinked)
            continue
        elif event.end == event.start:
            logging.info('{0}: skipped because zero duration'.format(format_time(event.start)))
            try:
                event.link_event(events[idx + 1])
            except IndexError:
                event.link_event(last_unlinked)
            continue

        # 链接开始和结束时间与其他事件相同的行
        # 假设脚本按开始时间排序，因此不需要搜索整个集合
        def same_start(x):
            return event.start == x.start
        processed = next((x for x in takewhile(same_start, reversed(events[:idx])) if not x.linked and x.end == event.end), None)
        if processed:
            event.link_event(processed)
        else:
            last_unlinked = event

    events = (e for e in events if not e.linked)

    search_groups = merge_short_lines_into_groups(events, chapter_times, max_ts_duration, max_ts_distance)

    # 将包含在其他组内的组链接到更大的组
    passed_groups = []
    for idx, group in enumerate(search_groups):
        try:
            other = next(x for x in reversed(search_groups[:idx])
                         if x[0].start <= group[0].start
                         and x[-1].end >= group[-1].end)
            for event in group:
                event.link_event(other[0])
        except StopIteration:
            passed_groups.append(group)
    return passed_groups


def calculate_shifts(src_stream, dst_stream, groups_list, normal_window, max_window, rewind_thresh):
    """
    计算每个事件组的时间偏移
    
    这是 Sushi 的核心算法，通过音频比较计算字幕时间偏移
    
    参数:
        src_stream: 源音频流
        dst_stream: 目标音频流
        groups_list: 事件组列表
        normal_window: 正常搜索窗口大小
        max_window: 最大搜索窗口大小
        rewind_thresh: 重新尝试前的连续错误阈值
    """
    def log_shift(state):
        logging.info('{0}-{1}: shift: {2:0.10f}, diff: {3:0.10f}'
                     .format(format_time(state["start_time"]), format_time(state["end_time"]), state["shift"], state["diff"]))

    def log_uncommitted(state, shift, left_side_shift, right_side_shift, search_offset):
        logging.debug('{0}-{1}: shift: {2:0.5f} [{3:0.5f}, {4:0.5f}], search offset: {5:0.6f}'
                      .format(format_time(state["start_time"]), format_time(state["end_time"]),
                              shift, left_side_shift, right_side_shift, search_offset))

    small_window = 1.5  # 小窗口大小，用于快速匹配
    idx = 0
    committed_states = []  # 已确认的状态
    uncommitted_states = []  # 未确认的状态
    window = normal_window
    while idx < len(groups_list):
        search_group = groups_list[idx]
        tv_audio = src_stream.get_substream(search_group[0].start, search_group[-1].end)
        original_time = search_group[0].start
        group_state = {"start_time": search_group[0].start, "end_time": search_group[-1].end, "shift": None, "diff": None}
        last_committed_shift = committed_states[-1]["shift"] if committed_states else 0
        diff = new_time = None

        if not uncommitted_states:
            if original_time + last_committed_shift > dst_stream.duration_seconds:
                # 事件超出音频范围，后面的事件也会失败
                for g in groups_list[idx:]:
                    committed_states.append({"start_time": g[0].start, "end_time": g[-1].end, "shift": None, "diff": None})
                    logging.info("{0}-{1}: outside of audio range".format(format_time(g[0].start), format_time(g[-1].end)))
                break

            if small_window < window:
                diff, new_time = dst_stream.find_substream(tv_audio, original_time + last_committed_shift, small_window)

            if new_time is not None and abs_diff(new_time - original_time, last_committed_shift) <= ALLOWED_ERROR:
                # 最快的情况 - 小窗口有效，立即提交组
                group_state.update({"shift": new_time - original_time, "diff": diff})
                committed_states.append(group_state)
                log_shift(group_state)
                if window != normal_window:
                    logging.info("Going back to window {0} from {1}".format(normal_window, window))
                    window = normal_window
                idx += 1
                continue

        # 将音频分成左右两半进行搜索，提高匹配精度
        left_audio_half, right_audio_half = np.split(tv_audio, [len(tv_audio[0]) // 2], axis=1)
        right_half_offset = len(left_audio_half[0]) / float(src_stream.sample_rate)
        terminate = False
        # 从上次确认的偏移开始搜索
        if original_time + last_committed_shift < dst_stream.duration_seconds:
            diff, new_time = dst_stream.find_substream(tv_audio, original_time + last_committed_shift, window)
            left_side_time = dst_stream.find_substream(left_audio_half, original_time + last_committed_shift, window)[1]
            right_side_time = dst_stream.find_substream(right_audio_half, original_time + last_committed_shift + right_half_offset, window)[1] - right_half_offset
            terminate = abs_diff(left_side_time, right_side_time) <= ALLOWED_ERROR and abs_diff(new_time, left_side_time) <= ALLOWED_ERROR
            log_uncommitted(group_state, new_time - original_time, left_side_time - original_time,
                            right_side_time - original_time, last_committed_shift)

        # 从上一个未确认状态的偏移开始再次尝试搜索
        if not terminate and uncommitted_states and uncommitted_states[-1]["shift"] is not None \
                and original_time + uncommitted_states[-1]["shift"] < dst_stream.duration_seconds:
            start_offset = uncommitted_states[-1]["shift"]
            diff, new_time = dst_stream.find_substream(tv_audio, original_time + start_offset, window)
            left_side_time = dst_stream.find_substream(left_audio_half, original_time + start_offset, window)[1]
            right_side_time = dst_stream.find_substream(right_audio_half, original_time + start_offset + right_half_offset, window)[1] - right_half_offset
            terminate = abs_diff(left_side_time, right_side_time) <= ALLOWED_ERROR and abs_diff(new_time, left_side_time) <= ALLOWED_ERROR
            log_uncommitted(group_state, new_time - original_time, left_side_time - original_time,
                            right_side_time - original_time, start_offset)

        if new_time is not None:
            shift = new_time - original_time
        else:
            shift = -1000000
            diff = 0.00001
            
        if not terminate:
            # 我们还没有回到正轨 - 将此组添加到未确认中
            group_state.update({"shift": shift, "diff": diff})
            uncommitted_states.append(group_state)
            idx += 1
            if rewind_thresh == len(uncommitted_states) and window < max_window:
                logging.warn("Detected possibly broken segment starting at {0}, increasing the window from {1} to {2}"
                             .format(format_time(uncommitted_states[0]["start_time"]), window, max_window))
                window = max_window
                idx = len(committed_states)
                del uncommitted_states[:]
            continue

        # 我们回到正轨了 - 将当前偏移应用到所有破损事件
        if uncommitted_states:
            logging.warning("Events from {0} to {1} will most likely be broken!".format(
                format_time(uncommitted_states[0]["start_time"]),
                format_time(uncommitted_states[-1]["end_time"])))

        uncommitted_states.append(group_state)
        for state in uncommitted_states:
            state.update({"shift": shift, "diff": diff})
            log_shift(state)
        committed_states.extend(uncommitted_states)
        del uncommitted_states[:]
        idx += 1

    # 记录所有未确认状态
    for state in uncommitted_states:
        log_shift(state)

    # 设置每个组的偏移
    for idx, (search_group, group_state) in enumerate(zip(groups_list, chain(committed_states, uncommitted_states))):
        if group_state["shift"] is None:
            # 链接到前面有效的事件
            for group in reversed(groups_list[:idx]):
                link_to = next((x for x in reversed(group) if not x.linked), None)
                if link_to:
                    for e in search_group:
                        e.link_event(link_to)
                    break
        else:
            for e in search_group:
                e.set_shift(group_state["shift"], group_state["diff"])


def check_file_exists(path, file_title):
    """
    检查文件是否存在
    
    参数:
        path: 文件路径
        file_title: 文件标题（用于错误消息）
        
    异常:
        SushiError: 当文件不存在时抛出
    """
    if path and not os.path.exists(path):
        raise SushiError("{0} file doesn't exist".format(file_title))


def format_full_path(temp_dir, base_path, postfix):
    """
    格式化完整文件路径
    
    参数:
        temp_dir: 临时目录
        base_path: 基础路径
        postfix: 后缀
        
    返回:
        str: 格式化后的完整路径
    """
    if temp_dir:
        return os.path.join(temp_dir, os.path.basename(base_path) + postfix)
    else:
        return base_path + postfix


def create_directory_if_not_exists(path):
    """
    如果目录不存在则创建
    
    参数:
        path: 目录路径
    """
    if path and not os.path.exists(path):
        os.makedirs(path)


def run(args):
    """
    主程序入口
    
    处理命令行参数，运行字幕同步流程
    
    参数:
        args: 命令行参数对象
    """
    ignore_chapters = args.chapters_file is not None and args.chapters_file.lower() == 'none'
    write_plot = plot_enabled and args.plot_path
    if write_plot:
        plt.clf()
        plt.ylabel('Shift, seconds')
        plt.xlabel('Event index')

    # 第一部分应该进行所有可能的验证，不应该花费大量时间
    check_file_exists(args.source, 'Source')
    check_file_exists(args.destination, 'Destination')
    check_file_exists(args.src_timecodes, 'Source timecodes')
    check_file_exists(args.dst_timecodes, 'Source timecodes')
    check_file_exists(args.script_file, 'Script')

    if not ignore_chapters:
        check_file_exists(args.chapters_file, 'Chapters')
    if args.src_keyframes not in ('auto', 'make'):
        check_file_exists(args.src_keyframes, 'Source keyframes')
    if args.dst_keyframes not in ('auto', 'make'):
        check_file_exists(args.dst_keyframes, 'Destination keyframes')

    if (args.src_timecodes and args.src_fps) or (args.dst_timecodes and args.dst_fps):
        raise SushiError('Both fps and timecodes file cannot be specified at the same time')

    src_demuxer = Demuxer(args.source)
    dst_demuxer = Demuxer(args.destination)

    if src_demuxer.is_wav and not args.script_file:
        raise SushiError("Script file isn't specified")

    if (args.src_keyframes and not args.dst_keyframes) or (args.dst_keyframes and not args.src_keyframes):
        raise SushiError('Either none or both of src and dst keyframes should be provided')

    create_directory_if_not_exists(args.temp_dir)

    # 选择源音频
    if src_demuxer.is_wav:
        src_audio_path = args.source
    else:
        src_audio_path = format_full_path(args.temp_dir, args.source, '.sushi.wav')
        src_demuxer.set_audio(stream_idx=args.src_audio_idx, output_path=src_audio_path, sample_rate=args.sample_rate)

    # 选择目标音频
    if dst_demuxer.is_wav:
        dst_audio_path = args.destination
    else:
        dst_audio_path = format_full_path(args.temp_dir, args.destination, '.sushi.wav')
        dst_demuxer.set_audio(stream_idx=args.dst_audio_idx, output_path=dst_audio_path, sample_rate=args.sample_rate)

    # 选择源字幕
    if args.script_file:
        src_script_path = args.script_file
    else:
        stype = src_demuxer.get_subs_type(args.src_script_idx)
        src_script_path = format_full_path(args.temp_dir, args.source, '.sushi' + stype)
        src_demuxer.set_script(stream_idx=args.src_script_idx, output_path=src_script_path)

    script_extension = get_extension(src_script_path)
    if script_extension not in ('.ass', '.srt'):
        raise SushiError('Unknown script type')

    # 选择目标字幕
    if args.output_script:
        dst_script_path = args.output_script
        dst_script_extension = get_extension(args.output_script)
        if dst_script_extension != script_extension:
            raise SushiError("Source and destination script file types don't match ({0} vs {1})"
                             .format(script_extension, dst_script_extension))
    else:
        dst_script_path = format_full_path(args.temp_dir, args.destination, '.sushi' + script_extension)

    # 选择章节
    if args.grouping and not ignore_chapters:
        if args.chapters_file:
            if get_extension(args.chapters_file) == '.xml':
                chapter_times = chapters.get_xml_start_times(args.chapters_file)
            else:
                chapter_times = chapters.get_ogm_start_times(args.chapters_file)
        elif not src_demuxer.is_wav:
            chapter_times = src_demuxer.chapters
            output_path = format_full_path(args.temp_dir, src_demuxer.path, ".sushi.chapters.txt")
            src_demuxer.set_chapters(output_path)
        else:
            chapter_times = []
    else:
        chapter_times = []

    # 选择关键帧和时间码
    if args.src_keyframes:
        def select_keyframes(file_arg, demuxer):
            auto_file = format_full_path(args.temp_dir, demuxer.path, '.sushi.keyframes.txt')
            if file_arg in ('auto', 'make'):
                if file_arg == 'make' or not os.path.exists(auto_file):
                    if not demuxer.has_video:
                        raise SushiError("Cannot make keyframes for {0} because it doesn't have any video!"
                                         .format(demuxer.path))
                    demuxer.set_keyframes(output_path=auto_file)
                return auto_file
            else:
                return file_arg

        def select_timecodes(external_file, fps_arg, demuxer):
            if external_file:
                return external_file
            elif fps_arg:
                return None
            elif demuxer.has_video:
                path = format_full_path(args.temp_dir, demuxer.path, '.sushi.timecodes.txt')
                demuxer.set_timecodes(output_path=path)
                return path
            else:
                raise SushiError('Fps, timecodes or video files must be provided if keyframes are used')

        src_keyframes_file = select_keyframes(args.src_keyframes, src_demuxer)
        dst_keyframes_file = select_keyframes(args.dst_keyframes, dst_demuxer)
        src_timecodes_file = select_timecodes(args.src_timecodes, args.src_fps, src_demuxer)
        dst_timecodes_file = select_timecodes(args.dst_timecodes, args.dst_fps, dst_demuxer)

    # 此时不应该再有任何失败，可以安全地开始耗时操作
    # 如运行实际的分离
    src_demuxer.demux()
    dst_demuxer.demux()

    try:
        if args.src_keyframes:
            src_timecodes = Timecodes.cfr(args.src_fps) if args.src_fps else Timecodes.from_file(src_timecodes_file)
            src_keytimes = [src_timecodes.get_frame_time(f) for f in keyframes.parse_keyframes(src_keyframes_file)]

            dst_timecodes = Timecodes.cfr(args.dst_fps) if args.dst_fps else Timecodes.from_file(dst_timecodes_file)
            dst_keytimes = [dst_timecodes.get_frame_time(f) for f in keyframes.parse_keyframes(dst_keyframes_file)]

        # 加载字幕文件
        script = AssScript.from_file(src_script_path) if script_extension == '.ass' else SrtScript.from_file(src_script_path)
        script.sort_by_time()

        # 加载音频流
        src_stream = WavStream(src_audio_path, sample_rate=args.sample_rate, sample_type=args.sample_type)
        dst_stream = WavStream(dst_audio_path, sample_rate=args.sample_rate, sample_type=args.sample_type)

        # 准备搜索组
        search_groups = prepare_search_groups(script.events,
                                              source_duration=src_stream.duration_seconds,
                                              chapter_times=chapter_times,
                                              max_ts_duration=args.max_ts_duration,
                                              max_ts_distance=args.max_ts_distance)

        # 计算时间偏移
        calculate_shifts(src_stream, dst_stream, search_groups,
                         normal_window=args.window,
                         max_window=args.max_window,
                         rewind_thresh=args.rewind_thresh if args.grouping else 0)

        events = script.events

        if write_plot:
            plt.plot([x.shift for x in events], label='From audio')

        # 根据分组设置处理事件
        if args.grouping:
            if not ignore_chapters and chapter_times:
                groups = groups_from_chapters(events, chapter_times)
                for g in groups:
                    fix_near_borders(g)
                    smooth_events([x for x in g if not x.linked], args.smooth_radius)
                groups = split_broken_groups(groups)
            else:
                fix_near_borders(events)
                smooth_events([x for x in events if not x.linked], args.smooth_radius)
                groups = detect_groups(events)

            if write_plot:
                plt.plot([x.shift for x in events], label='Borders fixed')

            # 记录每个组的偏移信息
            for g in groups:
                start_shift = g[0].shift
                end_shift = g[-1].shift
                avg_shift = average_shifts(g)
                logging.info('Group (start: {0}, end: {1}, lines: {2}), '
                             'shifts (start: {3}, end: {4}, average: {5})'
                             .format(format_time(g[0].start), format_time(g[-1].end), len(g), start_shift, end_shift,
                                     avg_shift))

            # 处理关键帧
            if args.src_keyframes:
                for e in (x for x in events if x.linked):
                    e.resolve_link()
                for g in groups:
                    snap_groups_to_keyframes(g, chapter_times, args.max_ts_duration, args.max_ts_distance, src_keytimes,
                                             dst_keytimes, src_timecodes, dst_timecodes, args.max_kf_distance, args.kf_mode)
        else:
            fix_near_borders(events)
            if write_plot:
                plt.plot([x.shift for x in events], label='Borders fixed')

            # 处理关键帧
            if args.src_keyframes:
                for e in (x for x in events if x.linked):
                    e.resolve_link()
                snap_groups_to_keyframes(events, chapter_times, args.max_ts_duration, args.max_ts_distance, src_keytimes,
                                         dst_keytimes, src_timecodes, dst_timecodes, args.max_kf_distance, args.kf_mode)

        # 应用偏移并保存结果
        for event in events:
            event.apply_shift()

        script.save_to_file(dst_script_path)

        # 绘制结果图表
        if write_plot:
            plt.plot([x.shift + (x._start_shift + x._end_shift) / 2.0 for x in events], label='After correction')
            plt.legend(fontsize=5, frameon=False, fancybox=False)
            plt.savefig(args.plot_path, dpi=300)

    finally:
        # 清理临时文件
        if args.cleanup:
            src_demuxer.cleanup()
            dst_demuxer.cleanup()
