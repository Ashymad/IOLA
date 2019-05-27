using IOLA
using MDCT
using DSP
using WAV
using Gnuplot

#period = 36.0 #mp3
#period = 68.0 #aac
#period = 17 #ac3?

codec = "aac"

params = Codec.getParams(Codec.AAC)
seg_hop = params.hop
transform = Codec.getTransformFunction(params)

bitrates = [128, 196, 256, 320]

diffs = Vector{Array{Float64, 2}}()

(y, fs, nbits, opt) = wavread("data/audio.wav")
seg_len = round(Int, fs/params.length)*params.length

push!(diffs, analyze(y[:,1], transform, seg_len, seg_hop))


for bitr in bitrates
    (y, fs, nbits, opt) = wavread("data/$codec/audio-$bitr.wav")
    push!(diffs, analyze(y[:,1], transform, seg_len, seg_hop))
end

@gp(:GP1, "set angles radians",
    "set polar",
    "set grid polar 15. lt -1 dt 0 lw 0.5",
    "unset border", "unset param",
    "unset xtics",
    2π*diffs[2][:,2]/seg_hop, diffs[2][:,1], "pt 7 ps 2 t '128k'",
    2π*diffs[3][:,2]/seg_hop, diffs[3][:,1], "pt 7 ps 2 t '196k'",
    2π*diffs[4][:,2]/seg_hop, diffs[4][:,1], "pt 7 ps 2 t '256k'",
    2π*diffs[5][:,2]/seg_hop, diffs[5][:,1], "pt 7 ps 2 t '320k'",
    2π*diffs[1][:,2]/seg_hop, diffs[1][:,1], "pt 7 ps 2 t 'none'"
   )

xs = 1:seg_hop
periodf = zeros(length(xs), 5)
for i = 1:length(xs), j = 1:5
    per = mod.(diffs[j][:,2],xs[i])
    sump = sum(per)
    periodf[i, j] = sum(abs.(per .- sump/length(per)))/sump
end

@gp(:GP2,
    xs, periodf[:,2], "w l tit '128k'",
    xs, periodf[:,3], "w l tit '196k'",
    xs, periodf[:,4], "w l tit '256k'",
    xs, periodf[:,5], "w l tit '320k'",
    xs, periodf[:,1], "w l tit 'none'"
   )
