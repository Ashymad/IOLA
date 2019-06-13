ENV["COLUMNS"]=72
import Base.GC.gc
gc()
using DSP.Periodograms: stft, nextfastfft
using DSP.Windows: hanning
using HDF5
using Base.Iterators: flatten, cycle, take, partition
using Statistics: mean, std
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
    win = hanning(nfft)
    for i in 1:length(idxs)
        st = abs.(stft(Float32.(data[:,idxs[i]][:]./typemax(Int16)), n, noverlap, window=win, nfft=nfft))
        X_batch[:, :, 1, i] = epsnormalise(20*log10.(max.(st, eps(Float32))))
    end
    Y_batch = labels[idxs[1]:idxs[end]] .+ 1
    return minibatch(X_batch, Y_batch, ceil(Int64, length(idxs) / minibatch_size))
end

minibatch_size = 4
batch_size = 16
train_size = 4096
test_size = 2048

train_batch_idxs = partition(1:train_size, batch_size)
test_batch_idxs = partition((train_size+1):(train_size+test_size), batch_size)

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

model = Chain(Conv(3, 3, 1, 16),
              Conv(3, 3, 16, 16),
              Conv(3, 3, 16, 16),
              Dense(309120, 512, pdrop=0.2),
              Dense(512, 256, pdrop=0.2),
              Dense(256, 4, identity, pdrop=0.2))

test_accs = Array{Float32, 1}()

h5open("./dataset.h5") do file
    data = file["data"]
    labels = file["labels"]

    for epoch in 1:10
        @info "Epoch: $epoch"

        for idx in progress(train_batch_idxs)
            let dtrn = make_batch(data, labels, idx, minibatch_size)
                for a in adam(model, dtrn); end
            end
            gc()
        end

        test_acc = 0
        for idx in progress(test_batch_idxs)
            let dtst = make_batch(data, labels, idx, minibatch_size)
                test_acc += accuracy(model, dtst)
            end
            gc()
        end
        test_acc /= length(test_batch_idxs)

        push!(test_accs, test_acc)

        @info "Test Accuracy: $test_acc"

    end

end


