ENV["COLUMNS"]=72
import Base.GC.gc
using DSP.Periodograms: stft, nextfastfft
using DSP.Windows: hanning
using HDF5
using Base.Iterators: flatten, cycle, take, partition
using Statistics: mean, std
using Random: shuffle!
using Knet
using Knet: Data, KnetArray
using LinearAlgebra: diag

function epsnormalise(x::AbstractArray{T}; dims=1, ϵ=eps(T)) where T
    μ = mean(x, dims = dims)
    σ = std(x, dims = dims, mean = μ, corrected=false)
    σ = map((x) -> abs(x) < ϵ ? ϵ : x, σ)
    return (x .- μ) ./ σ
end

function make_batch(data, labels, idxs, minibatch_size, cat_mapping; divider = 1, n = 512, noverlap = div(n, 2))
    nfft = nextfastfft(n)
    sz = div(div(size(data, 1), noverlap), divider) - 1
    X_batch = Array{Float32}(undef, div(nfft, 2) + 1, sz, 1, length(idxs))
    Y_batch = Array{UInt8}(undef, length(idxs))
    win = hanning(nfft)
    data_ind = 1:div(size(data, 1), divider)
    for i in 1:length(idxs)
        st = abs.(stft(Float32.(data[data_ind,idxs[i]][:]./typemax(Int16)), n, noverlap, window=win, nfft=nfft))
        X_batch[:, :, 1, i] = epsnormalise(20*log10.(max.(st, eps(Float32))))
        Y_batch[i] = cat_mapping[labels[idxs[i]][1]]
    end
    return minibatch(convert(KnetArray, X_batch), Y_batch, minibatch_size)
end

function confusion(model, data, no_categories, no_labels=no_categories)
    conf = zeros(no_categories, no_labels)
    sz = 0
    for mbatch in data
        pred = convert(Array, model(mbatch[1]))
        for ind = 1:length(mbatch[2])
            conf[findmax(pred[:, ind])[2], mbatch[2][ind]] += 1
        end
    end
    return conf
end

function squish_confusion(conf_ex, cat_mapping)
    vals = sort(collect(values(cat_mapping)))
    no_categories = length(unique(vals))
    conf = zeros(no_categories, no_categories)
    for i = 1:length(vals)
        conf[:, vals[i]] += conf_ex[:, i]
    end
    conf
end

function get_idxs(size, batch_size, cat_mapping, start_idx=1)
    idxs = Vector{Vector{Int}}()
    no_categories = length(unique(values(cat_mapping)))
    no_labels = length(cat_mapping)
    cat_lengths = zeros(no_categories)
    for val in values(cat_mapping)
        cat_lengths[val] += 1
    end
    cat_idxs = zeros(no_categories)

    batch_i = 1
    lab_i = start_idx
    push!(idxs, Vector{UInt}())
    while batch_i <= size/batch_size
        for i = 1:length(cat_lengths)
            push!(idxs[batch_i], lab_i + sum(cat_lengths[1:(i-1)]) + cat_idxs[i])
            cat_idxs[i] = mod(cat_idxs[i] + 1, cat_lengths[i])
            if length(idxs[batch_i]) == batch_size
                batch_i += 1
                if batch_i > size/batch_size
                    break
                else
                    push!(idxs, Vector{UInt}())
                end
            end
        end
        lab_i += no_labels
    end
    idxs
end

cat_mapping =
Dict{Int8, UInt8}(0  => 1,  # WAV
                  11 => 2,  # MP3 320
                  12 => 2,  # MP3 192
                  13 => 2,  # MP3 128
                  21 => 3,  # AAC 320
                  22 => 3,  # AAC 192
                  23 => 3,  # AAC 128
                  31 => 4,  # OGG 320
                  32 => 4,  # OGG 192
                  33 => 4,  # OGG 128
                  41 => 5,  # WMA 320
                  42 => 5,  # WMA 192
                  43 => 5,  # WMA 128
                  51 => 6,  # AC3 320
                  52 => 6,  # AC3 192
                  53 => 6,  # AC3 128
                 ) 

minibatch_size = 2
train_batch_size = 16
test_batch_size = 512
train_size = 5120
test_size = 2560

all_mapping =
Dict{Int8, UInt8}([(key, i) for (i, key) in enumerate(sort(collect(keys(cat_mapping))))])

no_categories = length(unique(values(cat_mapping)))
no_labels = length(cat_mapping)

train_batch_idxs = get_idxs(train_size, train_batch_size, cat_mapping)
test_batch_idxs = get_idxs(test_size, test_batch_size, cat_mapping, ceil(Int, train_batch_idxs[end][end]/no_labels)*no_labels + 1)

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

function train!(epochs, load=true, filename="model.jld2", train=true, test=true)
    gc()
    Knet.gc()

    model, test_confs = if load
        lf = Knet.load(filename)
        lf["model"], lf["test_confs"]
    else
        Chain(Conv(3, 3, 1, 16),
              Conv(3, 3, 16, 16),
              Conv(3, 3, 16, 16),
              Dense(102240, 300, pdrop=0.3),
              Dense(300, 300, pdrop=0.2),
              Dense(300, no_categories, identity, pdrop=0.2)),
        Array{Array{Float32, 2}, 1}()
    end
    best_acc = length(test_confs) > 0 ? sum(diag(test_confs[end]))/test_size : 0

    h5open("./dataset.h5") do file
        data = file["data"]
        labels = file["labels"]

        for epoch in 1:epochs
            @info "Epoch: $epoch"
            shuffle!(train_batch_idxs)
            if train
                for idx in progress(train_batch_idxs)
                    let dtrn = make_batch(data, labels, idx, minibatch_size, cat_mapping)
                        for a in adam(model, dtrn); end
                    end
                end
            end

            if test
                test_conf = zeros(no_categories, no_labels)
                for idx in progress(test_batch_idxs)
                    let dtst = make_batch(data, labels, idx, minibatch_size, all_mapping)
                        test_conf .+= confusion(model, dtst, no_categories, no_labels)
                    end
                end
                push!(test_confs, test_conf)

                acc = sum(diag(squish_confusion(test_conf, cat_mapping)))/test_size

                println("Confusion Matrix:")
                display(test_conf)
                println("Accuracy: $acc")
            else
                acc = best_acc + 1
            end
            if train
                if acc > best_acc
                    best_acc = acc
                    Knet.save(filename, "model", model, "test_confs", test_confs)
                end
            end
        end
    end

    return model, test_confs
end

