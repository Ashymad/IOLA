using Flux
using Flux: onecold, onehotbatch, ADAM, normalise
using Base.Iterators: partition
using Statistics
using DSP.Periodograms
import DSP.Periodograms.stft
using DSP.Windows
using HDF5
using Printf, BSON

function stft(s::AbstractArray{T, 4},n::Int=length(s)>>3, noverlap::Int=n>>1,
              psdonly::Union{Nothing,Periodograms.PSDOnly}=nothing;
              onesided::Bool=eltype(s)<:Real, nfft::Int=nextfastfft(n), fs::Real=1,
              window::Union{Function,AbstractVector,Nothing}=nothing) where T
    sz = div(size(s, 1), noverlap) - 1
    out = zeros(Complex{T}, div(nfft, 2) + 1, sz, 1, size(s, 4))
    for i = 1:size(s, 4)
        @views out[:,:,1,i] = stft(s[:,1,1,i][:], n, noverlap, psdonly; onesided=onesided, nfft=nfft, fs=fs, window=window)
    end
    out
end

function loss(x, y)
    ŷ = model(x)
    ce = sum(Flux.binarycrossentropy.(y, ŷ; ϵ=eps(Float32)))
    print("Loss: $ce")
    return ce
end

function make_minibatch(X, Y, idxs)
    noverlap = 256
    nfft = 512
    sz = div(size(X, 1), noverlap) - 1
    X_batch = Array{Float32}(undef, div(nfft, 2) + 1, sz, 1, length(idxs))
    win = hanning(nfft)
    for i in 1:length(idxs)
        st = abs.(stft(Float32.(X[:,idxs[i]][:]./typemax(Int16)), nfft, noverlap, window=win))
        X_batch[:, :, :, i] = normalise(20*log10.(map((x) -> x == 0 ? eps(Float32) : x, st)))
    end
    Y_batch = onehotbatch(Y[idxs[1]:idxs[end]], 0:3)
    return (X_batch, Y_batch)
end

model = Chain(
    Conv((3, 3), 1=>16, relu),
    MaxPool((2,2)),

    Conv((3, 3), 16=>16, relu),
    MaxPool((2,2)),

    Conv((3, 3), 16=>16, relu),
    MaxPool((2,2)),

    Conv((3, 3), 16=>16, relu),
    MaxPool((2,2)),

    x -> reshape(x, :, size(x, 4)),

    Dense(71904, 256),
    x -> relu.(x),
    Dropout(0.2),

    Dense(256, 256),
    x -> relu.(x),
    Dropout(0.2),

    Dense(256, 4),
    softmax
)

accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

batch_size = 8
train_size = 512
test_size = 128

mb_idxs = partition(1:train_size, batch_size)
mbs_idxs = partition((train_size+1):(train_size+test_size), batch_size)

@info("Loading data...")
train_set, test_set = h5open("./dataset.h5", "r") do file
    data = file["data"]
    labels = file["labels"]

    [make_minibatch(data, labels, i) for i in mb_idxs],
    [make_minibatch(data, labels, i) for i in mbs_idxs]
end

opt = ADAM(0.000001)

@info("Precompiling model")
model(train_set[1][1])

batches = length(train_set)

best_acc = 0.0
last_improvement = 0

@info("Starting training loop")
for epoch_idx in 1:100
    global best_acc, last_improvement
    ps = params(model)
    it = 0
    Flux.train!(loss, ps, train_set, opt, cb = () -> print(" Batch: $(it+=1)/$batches\r"))

    acc = 0
    for i = 1:length(test_set)
        acc += accuracy(test_set[i]...)
    end
    acc /= length(test_set)
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))

    # If our accuracy is good enough, quit out.
    if acc >= 0.999
        @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
        break
    end

    # If this is the best accuracy we've seen so far, save the model out
    if acc > best_acc
        @info(" -> New best accuracy! Saving model out to iola_conv.bson")
        BSON.@save "mnist_conv.bson" model epoch_idx acc
        best_acc = acc
        last_improvement = epoch_idx
    end

    # If we haven't seen improvement in 5 epochs, drop our learning rate:
    if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
        opt.eta /= 10.0
        @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

        # After dropping learning rate, give it a few epochs to improve
        last_improvement = epoch_idx
    end

    if epoch_idx - last_improvement >= 10
        @warn(" -> We're calling this converged.")
        break
    end
end
