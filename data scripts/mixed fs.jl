include("./Network.jl")
using DelimitedFiles
data = zeros(300, 1000000)
sizes = [100,1000,10000,100000,500000,1000000]
m = 9
for N in 1:(length(sizes))
    for iter in 1:50

        graph = mixed(m, sizes[N])
        temp_data = degree(graph)
        append!(temp_data, zeros(1000000-sizes[N]))

        data[(N-1)*50+iter,:] = temp_data
    end
end
writedlm( "mixed_data_fs.csv",  data, ',')