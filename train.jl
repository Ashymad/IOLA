using Flux
using Flux: onecold, onehotbatch, ADAM
using Base.Iterators: partition
using Statistics
using DSP.Periodograms
import DSP.Periodograms.stft
using DSP.Windows
using HDF5
using Printf, BSON
import Base.GC.gc

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

function epsnormalise(x::AbstractArray{T}; dims=1, ϵ=eps(T)) where T
    μ = mean(x, dims = dims)
    σ = std(x, dims = dims, mean = μ, corrected=false)
    σ = map((x) -> abs(x) < ϵ ? ϵ : x, σ)
    return (x .- μ) ./ σ
end

function loss(x, y, print_loss=true)
    ŷ = model(x)
    ce = sum(Flux.binarycrossentropy.(y, ŷ; ϵ=eps(Float32)))
    if print_loss
        print("Loss: $ce")
    end
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
        X_batch[:, :, :, i] = epsnormalise(20*log10.(max.(st, eps(Float32))))
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
function mbfn(fn, set)
    m = 0
    for i = 1:length(set)
        m += fn(set[i]...)
    end
    return m / length(set)
end

function make_batch(data, labels, idxs)
    idxs = idxs[1]:idxs[end]
    data = data[:,idxs]
    labels = labels[idxs]
    return [make_minibatch(data, labels, i) for i in minibatch_idxs]
end

minibatch_size = 8
batch_size = 512
train_size = 6144
test_size = 2048

minibatch_idxs = partition(1:batch_size, minibatch_size)
train_batch_idxs = partition(1:train_size, batch_size)
test_batch_idxs = partition((train_size+1):(train_size+test_size), batch_size)

opt = ADAM(0.000001)

batches = length(train_batch_idxs)
minibatches = length(minibatch_idxs)

best_acc = 0.0
last_improvement = 0

@info("Starting training loop")
for epoch_idx in 1:100
    global best_acc, last_improvement
    ps = params(model)
    b_it = 0
    test_acc = 0
    train_acc = 0
    train_loss = 0
    test_accs = Array{Float32, 1}()
    train_accs = Array{Float32, 1}()
    train_losses = Array{Float32, 1}()
    h5open("./dataset.h5", "r") do file
        data = file["data"]
        labels = file["labels"]

        for data_ind in train_batch_idxs
            mb_it = 0
            b_it += 1
            let train_set = make_batch(data, labels, data_ind)
                Flux.train!(loss, ps, train_set, opt, cb = () -> print(" Batch: $(b_it)/$batches Minibatch: $(mb_it+=1)/$minibatches                             \r"))
            end
            gc()
        end

        for data_ind in train_batch_idxs
            let train_set = make_batch(data, labels, data_ind)
                train_loss += mbfn((x,y) -> loss(x, y, false), train_set)
                train_acc += mbfn(accuracy, train_set)
            end
            gc()
        end
        train_acc /= length(train_batch_idxs)
        train_loss /= length(train_batch_idxs)

        for data_ind in test_batch_idxs
            let test_set = make_batch(data, labels, data_ind)
                test_acc += mbfn(accuracy, test_set)
            end
            gc()
        end
        test_acc /= length(test_batch_idxs)
    end


    @info(@sprintf("[%d]: Loss: %.4f Accuracy: %.4f Test accuracy: %.4f", epoch_idx, train_loss, train_acc, test_acc))
    push!(train_accs, train_acc)
    push!(train_losses, train_loss)
    push!(test_accs, test_acc)
    BSON.@save "iola_hist.bson" train_accs train_losses test_accs

    # If our accuracy is good enough, quit out.
    if test_acc >= 0.999
        @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
        break
    end

    # If this is the best accuracy we've seen so far, save the model out
    if test_acc > best_acc
        @info(" -> New best accuracy! Saving model out to iola_conv.bson")
        BSON.@save "mnist_conv.bson" model epoch_idx test_acc
        best_acc = test_acc
        last_improvement = epoch_idx
    end

    # If we haven't seen improvement in 5 epochs, drop our learning rate:
    if epoch_idx - last_improvement >= 5 && opt.eta > 1e-9
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
