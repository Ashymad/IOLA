using Flux
using HDF5

h5open("./dataset.h5", "r") do file
    data = file["data"]
    labels = file["labels"]
    
end
