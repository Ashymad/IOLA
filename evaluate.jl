using IOLA
using IOLA.Utils
using HDF5

max_points = 100
res_size = 2048
codecs_num = length(instances(Codec.CodecType))

parameters = Dict{Codec.CodecType, Tuple{Function,Int64,Int64}}()

function to_float64(arr::AbstractArray{T}) where T<:Integer
    return convert.(Float64, arr)./typemax(T)
end





h5open("dataset.h5") do dataset
    data = dataset["data"]
    siglen = size(data,1)
    po = max_points + 1
    @label loop1
    po -= 1
    prev_rpo = 0
    for codec in instances(Codec.CodecType)
        par = Codec.getparams(codec)
        len = (siglen-par.hop)/po
        len = round(len/par.length)*par.length
        rpo = div(siglen-par.hop,len)
        prev_rpo = prev_rpo == 0 ? rpo : prev_rpo
        if  prev_rpo != rpo
            @goto loop1
        end
    end
    reslen = Int(prev_rpo)
    println("Number of points: $reslen")

    for codec in instances(Codec.CodecType)
        par = Codec.getparams(codec)
        len = (siglen-par.hop)/po
        len = round(len/par.length)*par.length
        push!(parameters, codec => (Codec.gettransform(par), len, par.hop))
    end
    h5open("results.h5", "w") do results
        C = d_create(results, "C", datatype(Float64),
                     dataspace(codecs_num,res_size),
                     "chunk", (codecs_num,1))
        KD = d_create(results, "KD", datatype(Float64),
                      dataspace(reslen,2,codecs_num,res_size),
                      "chunk", (reslen,2,codecs_num,1))
        p = d_create(results, "p", datatype(Int64),
                     dataspace(codecs_num,res_size),
                     "chunk", (codecs_num,1))

        C_tmp = zeros(Float64,codecs_num)
        KD_tmp = zeros(Float64,reslen,2,codecs_num)
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
