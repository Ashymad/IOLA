from pydub import AudioSegment
from pydub.effects import normalize
import os
import numpy as np
from os.path import join, splitext
from random import randint
import h5py as h5

fragment_length = 30
maxsize = 0
fs = 44100
bd = 16

fmt = "mp3"
bitrates = [None, "320k", "192k", "128k"]
bitrates_labels = [0, 1, 2, 3]
lenbr = len(bitrates)

for (root, dirs, files) in os.walk(os.getcwd()):
    for cfile in files:
        if splitext(cfile)[-1].lower() == ".flac":
            maxsize += lenbr
actualsize = 0

with h5.File('dataset.h5', 'w') as f:
    labels = f.create_dataset("labels", (maxsize,), maxshape=(maxsize,),
                              dtype="int8")
    data = f.create_dataset("data", (maxsize, fragment_length*fs),
                            maxshape=(maxsize, fragment_length*fs),
                            dtype=f"int{bd}")
    for (root, dirs, files) in os.walk(os.getcwd()):
        try:
            for cfile in files:
                if splitext(cfile)[-1].lower() != ".flac":
                    continue
                fpath = join(root, cfile)
                print(f"Preparing: {fpath} ({int(actualsize/lenbr)}/{int(maxsize/lenbr)})")
                audio = AudioSegment\
                    .from_file(fpath, format="flac")
                if audio.duration_seconds > fragment_length and \
                   audio.frame_rate == fs:
                    audio = normalize(audio.set_channels(1).set_sample_width(bd//8))
                    start = randint(0, np.floor((audio.duration_seconds-fragment_length)*1000))
                    fragment = audio[start:(fragment_length*1000+start)]
                    for i in range(lenbr):
                        ind = i + actualsize
                        labels[ind] = bitrates_labels[i]
                        if bitrates[i] is not None:
                            fragment.export(f"/tmp/tmp.{fmt}", format=fmt,
                                            bitrate=bitrates[i])
                            fragment = AudioSegment.from_file(f"/tmp/tmp.{fmt}",
                                                              format=fmt)
                        data[ind, :] = fragment.get_array_of_samples()
                    actualsize += lenbr
        except KeyboardInterrupt:
            print("Received SIGINT, exiting...")
            break
        except Exception as ex:
            print(f"Exception: {ex}")
            continue
    labels.resize(actualsize, 0)
    data.resize(actualsize, 0)

