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

function epsnormalise(x::AbstractArray{T}; dims=1, ϵ=eps(T)) where T
    μ = mean(x, dims = dims)
    σ = std(x, dims = dims, mean = μ, corrected=false)
    σ = map((x) -> abs(x) < ϵ ? ϵ : x, σ)
    return (x .- μ) ./ σ
end

function make_batch(data, labels, idxs, minibatch_size; n = 512, noverlap = div(n, 2))
    nfft = nextfastfft(n)
    sz = div(size(data, 1), noverlap) - 1
    X_batch = Array{Float32}(undef, div(nfft, 2) + 1, sz, 1, length(idxs))
    Y_batch = Array{Int8}(undef, length(idxs))
    win = hanning(nfft)
    for i in 1:length(idxs)
        st = abs.(stft(Float32.(data[:,idxs[i]][:]./typemax(Int16)), n, noverlap, window=win, nfft=nfft))
        X_batch[:, :, 1, i] = epsnormalise(20*log10.(max.(st, eps(Float32))))
        Y_batch[i] = labels[idxs[i]][1] + 1
    end
    return minibatch(X_batch, Y_batch, minibatch_size)
end

function confusion(model, data, size)
    conf = zeros(Float32, size, size)
    for mbatch in data
        confl = zeros(Float32, size, size)
        pred = model(mbatch[1])
        for ind = 1:length(mbatch[2])
            confl[findmax(pred[ind, :])[2], mbatch[2][ind]] += 1
        end
        conf += confl ./ length(mbatch[2])
    end
    return conf ./ maximum(conf)
end

minibatch_size = 4
train_batch_size = 16
test_batch_size = 64
train_size = 512
test_size = 512

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

    model, test_accs = if load
        lf = Knet.load("model2.jld2")
        lf["model"], lf["test_accs"]
    else
        Chain(Conv(3, 3, 1, 16),
              Conv(3, 3, 16, 16),
              Conv(3, 3, 16, 16),
              Dense(309120, 256, pdrop=0.3),
              Dense(256, 256, pdrop=0.2),
              Dense(256, 4, identity, pdrop=0.2)),
        Array{Array{Float32, 1}, 1}()
    end

    h5open("./dataset.h5") do file
        data = file["data"]
        labels = file["labels"]

        for epoch in 1:epochs
            @info "Epoch: $epoch"
            shuffle!(train_batch_idxs)

            for idx in progress(train_batch_idxs)
                let dtrn = make_batch(data, labels, idx, minibatch_size)
                    for a in adam(model, dtrn); end
                end
                gc()
            end

            test_acc = zeros(4)
            for idx in progress(test_batch_idxs)
                for divo in 0:3
                    let dtst = make_batch(data, labels, filter((x) -> mod(x,4) == divo, idx), minibatch_size)
                        test_acc[divo+1] += accuracy(model, dtst)
                    end
                end
                gc()
            end
            test_acc /= length(test_batch_idxs)

            push!(test_accs, test_acc)

            @info "Test Accuracies: $test_acc, Mean: $(mean(test_acc))"
            if length(test_accs) > 1 && mean(test_acc) > maximum(mean.(test_accs[1:(end-1)]))
                Knet.save("model2.jld2", "model", model, "test_accs", test_accs)
            end
        end
    end

    return model, test_accs
end

