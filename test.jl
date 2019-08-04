using IOLA
using IOLA.Utils
using MDCT
using DSP
using WAV
using Gnuplot
using Statistics

codec = "aac"

params = Codec.getparams(Codec.AAC)
seg_hop = params.hop
transform = Codec.gettransform(params)

bitrates = [128, 196, 256, 320]

diffs = Vector{Array{Float64, 2}}()

(y, fs, nbits, opt) = wavread("data/audio.wav")
seg_len = round(Int, fs/params.length)*params.length

push!(diffs, analyze(y[:,1], transform, seg_len, seg_hop))

for bitr in bitrates
    (y, fs, nbits, opt) = wavread("data/$codec/audio-$bitr.wav")
    push!(diffs, analyze(y[:,1], transform, seg_len, seg_hop))
end

xs = 3:seg_hop
periodf = zeros(length(xs), 5)
for i = 1:length(xs), j = 1:5
    per = mod.(diffs[j][:,2],xs[i])
    periodf[i, j] = std(per; corrected=false)/mean(per)
end

@gp(:GP2,
    xs, periodf[:,2], "w l tit '128k'",
    xs, periodf[:,3], "w l tit '196k'",
    xs, periodf[:,4], "w l tit '256k'",
    xs, periodf[:,5], "w l tit '320k'",
    xs, periodf[:,1], "w l tit 'none'"
   )

period = zeros(5)
phis = zeros(length(diffs[1][:,2]), 5)
for i = 1:5
    period[i] = findperiod(diffs[i][:,2], xs)
    phis[:, i] = 2π*mod.(diffs[i][:,2], period[i])./period[i]
end

@gp(:GP1, "set angles radians",
    "set polar",
    "set grid polar 15. lt -1 dt 0 lw 0.5",
    "unset border", "unset param",
    "unset xtics",
    phis[:,2], diffs[2][:,1], "pt 7 ps 2 t '128k'",
    phis[:,3], diffs[3][:,1], "pt 7 ps 2 t '196k'",
    phis[:,4], diffs[4][:,1], "pt 7 ps 2 t '256k'",
    phis[:,5], diffs[5][:,1], "pt 7 ps 2 t '320k'",
    phis[:,1], diffs[1][:,1], "pt 7 ps 2 t 'none'"
   )

roms = zeros(5)
for i = 1:5
    roms[i] = radiusofmean(diffs[i][:,1], phis[:,i])
end
