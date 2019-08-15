from pydub import AudioSegment
import numpy as np
import h5py as h5

codecs_num = 16
mod_start = 10
fmt = ["wma", "ac3"]
ffmpeg_format = ["asf", "ac3"]
ffmpeg_codec = ["wmav2", "ac3"]
bitrates = ["320k", "192k", "128k"]

fs = 44100
bd = 16

with h5.File('dataset.h5', 'r+') as f:
    data = f["data"]
    seg_len = np.shape(data)[1]
    for i0 in np.arange(0, np.shape(data)[0], codecs_num):
        try:
            print(f"Progress: {i0}\r")
            segment = AudioSegment(
                data=bytearray(data[i0, :]),
                sample_width=bd//8,
                frame_rate=fs,
                channels=1)
            for i1 in range(len(fmt)):
                for i2 in range(len(bitrates)):
                    segment.export(f"/tmp/tmp.{fmt[i1]}",
                                   format=ffmpeg_format[i1],
                                   codec=ffmpeg_codec[i1],
                                   bitrate=bitrates[i2])
                    segment = AudioSegment.from_file(f"/tmp/tmp.{fmt[i1]}")
                    data[i0+mod_start+i1*len(bitrates)+i2, :] =\
                        segment.set_channels(1)\
                               .set_sample_width(bd//8)\
                               .get_array_of_samples()[0:seg_len]
        except KeyboardInterrupt:
            print("Received SIGINT, exiting...")
            break

