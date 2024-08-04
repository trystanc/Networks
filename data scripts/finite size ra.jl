include("./Network.jl")
using DelimitedFiles
data = zeros(360,1000000)
sizes = [100,1000,10000,100000,500000,1000000]
m = 5
for N in 1:(length(sizes))
    for iter in 1:60

        graph = rand_attach(m, sizes[N])
        temp_data = degree(graph)
        append!(temp_data, zeros(1000000-sizes[N]))

        data[(N-1)*60+iter,:] = temp_data
    end
end
writedlm( "RA_data_fs.csv",  data, ',')