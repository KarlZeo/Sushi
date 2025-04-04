import logging
import cv2
import numpy as np
from chunk import Chunk
import struct
import math
from time import time
import os.path

from .common import SushiError, clip
from functools import reduce

# WAV 格式常量定义
WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_EXTENSIBLE = 0xFFFE


class DownmixedWavFile(object):
    """
    WAV 文件读取类，支持将多声道音频下混为单声道
    处理 PCM 和 EXTENSIBLE 格式的 WAV 文件
    """
    _file = None

    def __init__(self, path):
        """
        初始化并打开 WAV 文件
        
        参数:
            path (str): WAV 文件路径
            
        异常:
            SushiError: 当文件格式不正确或不支持时抛出
        """
        super(DownmixedWavFile, self).__init__()
        self._file = open(path, 'rb')
        try:
            riff = Chunk(self._file, bigendian=False)
            if riff.getname() != b'RIFF':
                raise SushiError('File does not start with RIFF id')
            if riff.read(4) != b'WAVE':
                raise SushiError('Not a WAVE file')

            fmt_chunk_read = False
            data_chink_read = False
            file_size = os.path.getsize(path)

            # 逐个读取 WAV 文件中的区块
            while True:
                try:
                    chunk = Chunk(self._file, bigendian=False)
                except EOFError:
                    break

                if chunk.getname() == b'fmt ':
                    self._read_fmt_chunk(chunk)  # 读取格式区块
                    fmt_chunk_read = True
                elif chunk.getname() == b'data':
                    if file_size > 0xFFFFFFFF:
                        # 处理大型 WAV 文件（超过 4GB 的情况）
                        self.frames_count = (file_size - self._file.tell()) // self.frame_size
                    else:
                        self.frames_count = chunk.chunksize // self.frame_size
                    data_chink_read = True
                    break
                chunk.skip()
            if not fmt_chunk_read or not data_chink_read:
                raise SushiError('Invalid WAV file')
        except Exception:
            self.close()
            raise

    def __del__(self):
        """析构函数，确保文件被关闭"""
        self.close()

    def close(self):
        """关闭打开的文件句柄"""
        if self._file:
            self._file.close()
            self._file = None

    def readframes(self, count):
        """
        读取指定数量的音频帧并下混为单声道
        
        参数:
            count (int): 要读取的帧数
            
        返回:
            numpy.ndarray: 下混后的音频数据，float32 格式
        """
        if not count:
            return ''
        data = self._file.read(count * self.frame_size)
        
        # 根据采样宽度解析字节数据
        if self.sample_width == 2:
            # 16位采样
            unpacked = np.fromstring(data, dtype=np.int16)
        elif self.sample_width == 3:
            # 24位采样，需要特殊处理
            raw_bytes = np.ndarray(len(data), 'int8', data)
            unpacked = np.zeros(len(data) // 3, np.int16)
            unpacked.view(dtype='int8')[0::2] = raw_bytes[1::3]
            unpacked.view(dtype='int8')[1::2] = raw_bytes[2::3]
        else:
            raise SushiError('Unsupported sample width: {0}'.format(self.sample_width))

        # 转换为浮点数据
        unpacked = unpacked.astype('float32')

        # 如果是单声道，直接返回
        if self.channels_count == 1:
            return unpacked
        else:
            # 多声道下混为单声道
            min_length = len(unpacked) // self.channels_count
            real_length = len(unpacked) / float(self.channels_count)
            if min_length != real_length:
                logging.error("Length of audio channels didn't match. This might result in broken output")

            # 提取各声道并求平均
            channels = (unpacked[i::self.channels_count] for i in range(self.channels_count))
            data = reduce(lambda a, b: a[:min_length] + b[:min_length], channels)
            data /= float(self.channels_count)
            return data

    def _read_fmt_chunk(self, chunk):
        """
        读取 WAV 文件的 fmt 区块，获取音频格式信息
        
        参数:
            chunk (Chunk): fmt 区块对象
        
        异常:
            SushiError: 当格式不支持时抛出
        """
        wFormatTag, self.channels_count, self.framerate, dwAvgBytesPerSec, wBlockAlign = struct.unpack('<HHLLH',
                                                                                                       chunk.read(14))
        if wFormatTag == WAVE_FORMAT_PCM or wFormatTag == WAVE_FORMAT_EXTENSIBLE:  # ignore the rest
            bits_per_sample = struct.unpack('<H', chunk.read(2))[0]
            self.sample_width = (bits_per_sample + 7) // 8
        else:
            raise SushiError('unknown format: {0}'.format(wFormatTag))
        self.frame_size = self.channels_count * self.sample_width


class WavStream(object):
    """
    WAV 音频流处理类，提供音频重采样和子流搜索功能
    """
    READ_CHUNK_SIZE = 1  # 每次读取一秒数据，效率最高
    PADDING_SECONDS = 10  # 在音频两端添加的填充秒数

    def __init__(self, path, sample_rate=12000, sample_type='uint8'):
        """
        初始化 WavStream 对象
        
        参数:
            path (str): WAV 文件路径
            sample_rate (int): 目标采样率，默认 12000Hz
            sample_type (str): 采样数据类型，'uint8' 或 'float32'
            
        异常:
            SushiError: 当采样类型不支持或加载文件失败时抛出
        """
        if sample_type not in ('float32', 'uint8'):
            raise SushiError('Unknown sample type of WAV stream, must be uint8 or float32')

        stream = DownmixedWavFile(path)
        total_seconds = stream.frames_count / float(stream.framerate)
        downsample_rate = sample_rate / float(stream.framerate)

        self.sample_count = math.ceil(total_seconds * sample_rate)
        self.sample_rate = sample_rate
        # 预分配数据数组，包括两端的填充空间
        self.data = np.empty((1, int(self.PADDING_SECONDS * 2 * stream.framerate + self.sample_count)), np.float32)
        self.padding_size = 10 * stream.framerate
        before_read = time()
        try:
            seconds_read = 0
            samples_read = self.padding_size
            # 逐块读取音频数据
            while seconds_read < total_seconds:
                data = stream.readframes(int(self.READ_CHUNK_SIZE * stream.framerate))
                new_length = int(round(len(data) * downsample_rate))

                dst_view = self.data[0][samples_read:samples_read + new_length]

                # 如果需要重采样
                if downsample_rate != 1:
                    data = data.reshape((1, len(data)))
                    data = cv2.resize(data, (new_length, 1), interpolation=cv2.INTER_NEAREST)[0]

                np.copyto(dst_view, data, casting='no')
                samples_read += new_length
                seconds_read += self.READ_CHUNK_SIZE

            # 对音频两端进行填充
            self.data[0][0:self.padding_size].fill(self.data[0][self.padding_size])
            self.data[0][-self.padding_size:].fill(self.data[0][-self.padding_size - 1])

            # 归一化处理
            # 通过中值的 3 倍对音频进行截断，减少异常值的影响
            max_value = np.median(self.data[self.data >= 0], overwrite_input=True) * 3
            min_value = np.median(self.data[self.data <= 0], overwrite_input=True) * 3

            np.clip(self.data, min_value, max_value, out=self.data)

            self.data -= min_value
            self.data /= (max_value - min_value)

            # 如果需要 uint8 格式，转换为 0-255 范围
            if sample_type == 'uint8':
                self.data *= 255.0
                self.data += 0.5
                self.data = self.data.astype('uint8')

        except Exception as e:
            raise SushiError('Error while loading {0}: {1}'.format(path, e))
        finally:
            stream.close()
        logging.info('Done reading WAV {0} in {1}s'.format(path, time() - before_read))

    @property
    def duration_seconds(self):
        """
        获取音频流的持续时间（秒）
        
        返回:
            float: 音频持续时间
        """
        return self.sample_count / self.sample_rate

    def get_substream(self, start, end):
        """
        获取指定时间范围的子音频流
        
        参数:
            start (float): 开始时间（秒）
            end (float): 结束时间（秒）
            
        返回:
            numpy.ndarray: 子音频流数据
        """
        start_off = self._get_sample_for_time(start)
        end_off = self._get_sample_for_time(end)
        return self.data[:, start_off:end_off]

    def _get_sample_for_time(self, timestamp):
        """
        将时间转换为样本索引，考虑填充
        
        参数:
            timestamp (float): 时间（秒）
            
        返回:
            int: 对应的样本索引
        """
        # 此函数获取时间对应的实际样本索引，考虑填充
        return int(self.sample_rate * timestamp) + self.padding_size

    def find_substream(self, pattern, window_center, window_size):
        """
        在音频流中查找匹配的模式
        
        参数:
            pattern (numpy.ndarray): 要查找的模式
            window_center (float): 搜索窗口中心时间（秒）
            window_size (float): 搜索窗口半径（秒）
            
        返回:
            tuple: (相似度, 匹配位置的时间)，相似度越低表示匹配越好
        """
        start_time = clip(window_center - window_size, -self.PADDING_SECONDS, self.duration_seconds)
        end_time = clip(window_center + window_size, 0, self.duration_seconds + self.PADDING_SECONDS)

        start_sample = self._get_sample_for_time(start_time)
        end_sample = self._get_sample_for_time(end_time) + len(pattern[0])

        search_source = self.data[:, start_sample:end_sample]
        # 使用 OpenCV 的模板匹配算法查找模式
        result = cv2.matchTemplate(search_source, pattern, cv2.TM_SQDIFF_NORMED)
        min_idx = result.argmin(axis=1)[0]

        # 返回最佳匹配的相似度和时间位置
        return result[0][min_idx], start_time + (min_idx / float(self.sample_rate))
