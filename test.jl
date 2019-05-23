using IOLA
using MDCT
using DSP
using WAV
using Gnuplot

#period = 36.0 #mp3
period = 68.0 #aac

codec = "ogg"

params = Codec.getParams(Codec.OGG)
hop = params.hop
transform = Codec.getTransformFunction(params)

(y, fs, nbits, opt) = wavread("data/audio.wav")

diffs = analyze(y[1:div(end,2),1], transform, convert(Int, fs), hop)

(y, fs, nbits, opt) = wavread("data/$codec/audio-128.wav")

diffs128 = analyze(y[1:div(end,2),1], transform, convert(Int, fs), hop)

(y, fs, nbits, opt) = wavread("data/$codec/audio-196.wav")

diffs196 = analyze(y[1:div(end,2),1], transform, convert(Int, fs), hop)

(y, fs, nbits, opt) = wavread("data/$codec/audio-256.wav")

diffs256 = analyze(y[1:div(end,2),1], transform, convert(Int, fs), hop)

(y, fs, nbits, opt) = wavread("data/$codec/audio-320.wav")

diffs320 = analyze(y[1:div(end,2),1], transform, convert(Int, fs), hop)

@gp(:GP1, "set angles radians",
    "set polar",
    "set grid polar 15. lt -1 dt 0 lw 0.5",
    "unset border", "unset param",
    "unset xtics",
    2π*mod.(diffs128[:,2], period)/period, diffs128[:,1], "pt 7 ps 2 t '128'",
    2π*mod.(diffs196[:,2], period)/period, diffs196[:,1], "pt 7 ps 2 t '196'",
    2π*mod.(diffs256[:,2], period)/period, diffs256[:,1], "pt 7 ps 2 t '256'",
    2π*mod.(diffs320[:,2], period)/period, diffs320[:,1], "pt 7 ps 2 t '320'",
    2π*mod.(diffs[:,2], period)/period, diffs[:,1], "pt 7 ps 2 t 'none'"
   )

xs = 1:hop/2
periodf = zeros(length(xs))
for i = eachindex(periodf)
    per = mod.(diffs128[:,2],xs[i])
    periodf[i] = sum(abs.(per .- sum(per)/length(per)))
end

@gp :GP2 xs periodf "w l tit 'Period'"
