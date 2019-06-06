using Flux
using DSP.Periodograms
import DSP.Periodograms.stft
using DSP.Windows
using HDF5

function stft(s::AbstractArray{T, 4},n::Int=length(s)>>3, noverlap::Int=n>>1,
              psdonly::Union{Nothing,Periodograms.PSDOnly}=nothing;
              onesided::Bool=eltype(s)<:Real, nfft::Int=nextfastfft(n), fs::Real=1,
              window::Union{Function,AbstractVector,Nothing}=nothing) where T
    sz = div(size(s, 1), noverlap) - 1
    out = zeros(Complex{T}, div(nfft, 2) + 1, sz, 1, size(s, 4))
    for i = 1:size(s, 4)
        out[:,:,1,i] = stft(s[:,1,1,i][:], n, noverlap, psdonly; onesided=onesided, nfft=nfft, fs=fs, window=window)
    end
    out
end

m = Chain(
    x -> abs.(stft(x, 512, fs=44100, window=hanning(512))),
    Flux.normalise,

    Conv((3, 3), 1=>16, relu),
    MaxPool((2,2)),

    Conv((3, 3), 16=>32, relu),
    MaxPool((2,2)),

    Conv((3, 3), 32=>48, relu),
    MaxPool((2,2)),

    Conv((3, 3), 48=>64, relu),
    MaxPool((2,2)),

    x -> reshape(x, :, size(x, 4)),

    Dense(287616, 256, relu),
    Dropout(0.2),

    Dense(256, 256, relu),
    Dropout(0.2),

    Dense(256, 4),
    softmax
)



test = h5open("./dataset.h5", "r") do file
    data = file["data"]
    labels = file["labels"]

    data[:,1]
end
test = reshape(test, :, 1, 1, 1)/typemax(Int16)
