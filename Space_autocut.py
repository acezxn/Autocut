from pydub import AudioSegment
from pydub.playback import play
import array
import pydub
import numpy as np
from pydub.utils import get_array_type
from tqdm import tqdm
import sys

file_name = sys.argv[1]

startMin = 0
startSec = 0

endMin = 0
endSec = 26

ignore = 2000

ignoring = False
# Time to miliseconds
startTime = startMin*60*1000+startSec*1000
endTime = endMin*60*1000+endSec*1000

# Opening file and extracting segment
audiofile = AudioSegment.from_mp3(file_name)

def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")


# extract = audiofile[startTime:endTime]
extract = audiofile
extract.export( '../Documents/sound/test.mp3', format="mp3")
sr, numeric_array = read('../Documents/sound/test.mp3')

arr = []
# finding average amplitide

print(len(numeric_array))
def mvavg_calc(numeric_array, window_size):
    moving_avg = []
    window_size -= 1
    print('calculating moving average')
    for j in tqdm(range(len(numeric_array))):
        # if (j % 1000 == 0):
        #     print(j/ (len(numeric_array) - 1))
        s = []
        if j + window_size > len(numeric_array)-1:
            window_size = 0
        for i in range(j, j+window_size+1):
            # print(j, j+window_size)
            # s.append([abs(numeric_array[i][0]), abs(numeric_array[i][1])])
            s.append([numeric_array[i][0], numeric_array[i][1]])
        sm = [0, 0]
        for i in s:
            sm[0] += i[0]
            sm[1] += i[1]
        avg = [sm[0]/ (len(s)), sm[1] / (len(s))]
        moving_avg.append(avg)
    print('calc_complete')
    return moving_avg
def avg_calc(numeric_array):
    s = []
    for i in numeric_array:
        s.append([abs(i[0]), abs(i[1])])
    sm = [0, 0]
    print('adding amps')
    for i in tqdm(s):
        sm[0] += i[0]
        sm[1] += i[1]
    ref = [sm[0]/ (len(numeric_array)), sm[1] / (len(numeric_array))]
    return ref

ref = avg_calc(numeric_array)
avg = mvavg_calc(numeric_array, 5)
print(ref)

count = 0
numeric_list = numeric_array.tolist()
idx = 0
for i in numeric_array:
    if ignoring:
        count += 1
        if (count) >= ignore:
            count = 0
            ignoring = False
            idx += 1
            continue
        arr.append(i)
        idx += 1
        continue
    if avg[idx][0] > ref[0] and avg[idx][1] > ref[1]:
        while count < ignore:
            count += 1
        count = 0
        arr.append(i)
        ignoring = True
        idx += 1
        continue
    idx += 1

a = np.array(arr)
print(a)
write('out.mp3', sr, a)
