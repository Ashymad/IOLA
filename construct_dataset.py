from pydub import AudioSegment
from pydub.effects import normalize
import os
import numpy as np
from os.path import join, splitext
from random import randint, shuffle
import h5py as h5

fragment_length = 10
max_dataset_size = 65536
fs = 44100
bd = 16

fmt = [None, "mp3", "aac", "ogg", "wma", "ac3"]
ffmpeg_format = [None, "mp3", "adts", "ogg", "asf", "ac3"]
ffmpeg_codec = [None, "libmp3lame", "aac", "libvorbis", "wmav2", "ac3"]
bitrates = ["320k", "192k", "128k"]
bitrates_labels = [1, 2, 3]
fmt_labels = [0, 10, 20, 30, 40, 50]
lenbr = len(bitrates)*len(fmt) - len(bitrates) + 1

allfiles = []

for (root, dirs, files) in os.walk(os.getcwd()):
    for cfile in files:
        if splitext(cfile)[-1].lower() == ".flac":
            allfiles.append(join(root, cfile))

maxsize = len(allfiles)*lenbr
shuffle(allfiles)
actualsize = 0

with h5.File('dataset.h5', 'w') as f:
    labels = f.create_dataset("labels", (maxsize,), maxshape=(maxsize,),
                              dtype="int8")
    data = f.create_dataset("data", (maxsize, fragment_length*fs),
                            maxshape=(maxsize, fragment_length*fs),
                            dtype=f"int{bd}")
    for fpath in allfiles:
        try:
            audio = AudioSegment\
                .from_file(fpath, format="flac")
            if audio.duration_seconds > fragment_length and \
               audio.frame_rate == fs:
                print(f"Preparing: {fpath} ({int(actualsize/lenbr)}/{int(maxsize/lenbr)})")
                audio = normalize(audio.set_sample_width(bd//8))
                start = randint(0, np.floor((audio.duration_seconds-fragment_length)*1000))
                fragment = audio[start:(fragment_length*1000+start)]
                for i1 in range(len(fmt)):
                    if fmt[i1] is None:
                        labels[actualsize] = fmt_labels[i1]
                        data[actualsize, :] = fragment.set_channels(1).get_array_of_samples()
                        actualsize += 1
                    else:
                        for i2 in range(len(bitrates)):
                            labels[actualsize] = bitrates_labels[i2] + fmt_labels[i1]
                            fragment.export(f"/tmp/tmp.{fmt[i1]}",
                                            format=ffmpeg_format[i1],
                                            codec=ffmpeg_codec[i1],
                                            bitrate=bitrates[i2])
                            fragment = AudioSegment.from_file(f"/tmp/tmp.{fmt[i1]}")
                            data[actualsize, :] = fragment.set_channels(1).get_array_of_samples()[0:fragment_length*fs]
                            actualsize += 1
            if actualsize >= max_dataset_size:
                break
        except KeyboardInterrupt:
            print("Received SIGINT, exiting...")
            break
        except Exception as ex:
            print(f"Exception: {ex}")
            continue
    labels.resize(actualsize, 0)
    data.resize(actualsize, 0)

