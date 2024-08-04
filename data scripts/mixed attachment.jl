include("./Network.jl")
using DelimitedFiles
N = 500000
data = zeros(40,N)
ms = [3,9,27,81]

for m in 1:4
    for n in 1:10
        println((m-1)*10+n)
        graph = mixed(ms[m], N)

        data[(m-1)*10+n,:] = degree(graph)
    end
end
writedlm("mixed_data_1.csv",  data, ',')