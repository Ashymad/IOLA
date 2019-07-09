ENV["COLUMNS"]=72
import Base.GC.gc
gc()
using DSP.Periodograms: stft, nextfastfft
using DSP.Windows: hanning
using HDF5
using Base.Iterators: flatten, cycle, take, partition
using Statistics: mean, std
using Random: shuffle!
using Knet
using Knet: Data
using LinearAlgebra: diag

function epsnormalise(x::AbstractArray{T}; dims=1, ϵ=eps(T)) where T
    μ = mean(x, dims = dims)
    σ = std(x, dims = dims, mean = μ, corrected=false)
    σ = map((x) -> abs(x) < ϵ ? ϵ : x, σ)
    return (x .- μ) ./ σ
end

function make_batch(data, labels, idxs, minibatch_size; divider = 1, n = 512, noverlap = div(n, 2))
    nfft = nextfastfft(n)
    sz = div(div(size(data, 1), noverlap), divider) - 1
    X_batch = Array{Float32}(undef, div(nfft, 2) + 1, sz, 1, length(idxs))
    Y_batch = Array{Int8}(undef, length(idxs))
    win = hanning(nfft)
    data_ind = 1:div(size(data, 1), divider)
    for i in 1:length(idxs)
        st = abs.(stft(Float32.(data[data_ind,idxs[i]][:]./typemax(Int16)), n, noverlap, window=win, nfft=nfft))
        X_batch[:, :, 1, i] = epsnormalise(20*log10.(max.(st, eps(Float32))))
        Y_batch[i] = labels[idxs[i]][1] + 1
    end
    return minibatch(X_batch, Y_batch, minibatch_size)
end

function confusion(model, data, size)
    conf = zeros(Float32, size, size)
    for mbatch in data
        pred = model(mbatch[1])
        for ind = 1:length(mbatch[2])
            conf[findmax(pred[ind, :])[2], mbatch[2][ind]] += 1
        end
    end
    return conf ./ length(data)
end

const no_categories = 4

const minibatch_size = 4
const train_batch_size = 256
const test_batch_size = 512
const train_size = 2048
const test_size = 2048

train_batch_idxs = collect(partition(1:train_size, train_batch_size))
test_batch_idxs = partition((train_size+1):(train_size+test_size), test_batch_size)

struct Conv; w; b; f; p; end
(c::Conv)(x) = c.f.(pool(conv4(c.w, dropout(x,c.p)) .+ c.b))
Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu;pdrop=0) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f, pdrop)

struct Dense; w; b; f; p; end
(d::Dense)(x) = d.f.(d.w * mat(dropout(x,d.p)) .+ d.b) # mat reshapes 4-D tensor to 2-D matrix so we can use matmul
Dense(i::Int,o::Int,f=relu;pdrop=0) = Dense(param(o,i), param0(o), f, pdrop)

struct Chain; layers; Chain(layers...) = new(layers); end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)
(c::Chain)(d::Data) = mean(c(x,y) for (x,y) in d)

function train!(epochs, load=true)

    model, test_confs = if load
        lf = Knet.load("model2.jld2")
        lf["model"], lf["test_confs"]
    else
        Chain(Conv(3, 3, 1, 16),
              Conv(3, 3, 16, 16),
              Conv(3, 3, 16, 16),
              Dense(50400, 256, pdrop=0.3),
              Dense(256, 256, pdrop=0.2),
              Dense(256, no_categories, identity, pdrop=0.2)),
        Array{Array{Float32, 2}, 1}()
    end
    best_acc = length(test_confs) > 0 ? mean(diag(test_confs[end])) : 0

    h5open("./dataset.h5") do file
        data = file["data"]
        labels = file["labels"]

        for epoch in 1:epochs
            @info "Epoch: $epoch"
            shuffle!(train_batch_idxs)

            for idx in progress(train_batch_idxs)
                let dtrn = make_batch(data, labels, idx, minibatch_size, divider=6)
                    for a in adam(model, dtrn); end
                end
                gc()
            end

            test_conf = zeros(no_categories, no_categories)
            for idx in progress(test_batch_idxs)
                let dtst = make_batch(data, labels, idx, minibatch_size, divider=6)
                    test_conf .+= confusion(model, dtst, no_categories)
                end
                gc()
            end
            test_conf ./= length(test_batch_idxs)

            push!(test_confs, test_conf)

            acc = mean(diag(test_conf))

            println("Confusion Matrix:")
            display(test_conf)
            println("Accuracy: $acc")
            if acc > best_acc
                best_acc = acc
                Knet.save("model2.jld2", "model", model, "test_confs", test_confs)
            end
        end
    end

    return model, test_confs
end

