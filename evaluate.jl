using IOLA
using IOLA.Utils
using MDCT
using HDF5
using Statistics

fs = 44100
seg_len = 1
res_size = 4096
codecs_num = length(instances(Codec.CodecType))

parameters = Dict{Codec.CodecType, Tuple{Function,Int64,Int64}}()

function to_float64(arr::AbstractArray{T}) where T<:Integer
    return convert.(Float64, arr)./typemax(T)
end

for codec in instances(Codec.CodecType)
    par = Codec.getparams(codec)
    len = ceil(Int64, seg_len*fs/par.length)*par.length
    push!(parameters, codec => (Codec.gettransform(par), len, par.hop))
end

h5open("dataset.h5") do dataset
    data = dataset["data"]
    h5open("results.h5", "w") do results
        KD_size = Int(size(data,1)/fs - 1)
        C = d_create(results, "C", datatype(Float64),
                     dataspace(codecs_num,res_size),
                     "chunk", (codecs_num,1))
        KD = d_create(results, "KD", datatype(Float64),
                      dataspace(KD_size,2,codecs_num,res_size),
                      "chunk", (KD_size,2,codecs_num,1))
        p = d_create(results, "p", datatype(Int64),
                     dataspace(codecs_num,res_size),
                     "chunk", (codecs_num,1))

        C_tmp = zeros(Float64,codecs_num)
        KD_tmp = zeros(Float64,KD_size,2,codecs_num)
        p_tmp = zeros(Int64,codecs_num)
        
        for data_i in 1:res_size
            print("Progress: $data_i/$res_size\r")
            fdata = to_float64(view(data[:,data_i],:,1))
            for (i,codec) in enumerate(instances(Codec.CodecType))
                KD_tmp[:,:,i] = analyze(fdata, parameters[codec]...)
                p_tmp[i] = findperiod(KD_tmp[:,2,i], 3:parameters[codec][3])
                C_tmp[i] = radiusofmean(KD_tmp[:,1,i], 2π*mod.(KD_tmp[:,2,i],
                                                      p_tmp[i])./p_tmp[i])
            end
            C[:,data_i] = C_tmp
            KD[:,:,:,data_i] = KD_tmp
            p[:,data_i] = p_tmp
        end
    end
end
