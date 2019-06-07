using Flux
using Flux: onecold, onehotbatch, ADAM, logitcrossentropy, normalise
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



test = h5open("./dataset.h5", "r") do file
    data = file["data"]
    labels = file["labels"]

    model = Chain(
        Conv((3, 3), 1=>16, relu),
        MaxPool((2,2)),

        Conv((3, 3), 16=>32, relu),
        MaxPool((2,2)),

        Conv((3, 3), 32=>32, relu),
        MaxPool((2,2)),

        Conv((3, 3), 32=>32, relu),
        MaxPool((2,2)),

        x -> reshape(x, :, size(x, 4)),

        Dense(143808, 256),
        Dropout(0.2),

        Dense(256, 256),
        Dropout(0.2),

        Dense(256, 4),
        softmax
    )

    function loss(x::Array{T}, y, ε = eps(1.0)) where T
        ŷ = model(x)
        c = Array{T}(undef, size(y)...)
        for it = 1:length(c)
            c[it] = y[it] > 0.5 ? -(log(ŷ[it]+ε)) : -(log((1.0-ŷ[it])+ε))
        end
        return c
    end

    function make_minibatch(X, Y, idxs)
        noverlap = 256
        nfft = 512
        sz = div(size(X, 1), noverlap) - 1
        X_batch = Array{Float32}(undef, div(nfft, 2) + 1, sz, 1, length(idxs))
        for i in 1:length(idxs)
            X_batch[:, :, :, i] = normalise(abs.(stft(Float32.(X[:,idxs[i]]./typemax(Int16))[:], nfft, noverlap)))
        end
        Y_batch = onehotbatch(Y[idxs[1]:idxs[end]], 0:3)
        return (X_batch, Y_batch)
    end

    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
    batch_size = 8
    train_size = 512
    test_size = 128

    mb_idxs = partition(1:train_size, batch_size)
    mbs_idxs = partition((train_size+1):(train_size+test_size), batch_size)
    @info("Loading data...")
    train_set = [make_minibatch(data, labels, i) for i in mb_idxs]
    test_set = [make_minibatch(data, labels, i) for i in mbs_idxs]

    opt = ADAM(0.001)

    @info("Precompiling model")
    model(train_set[1][1])

    batches = length(train_set)

    best_acc = 0.0
    last_improvement = 0
    @info("Starting training loop")
    for epoch_idx in 1:100
        ps = params(model)
        it = 0
        Flux.train!(loss, ps, train_set, opt, cb = () -> print("Batch: $(it+=1)/$batches\r"))

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
            @info(" -> New best accuracy! Saving model out to mnist_conv.bson")
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
end
