include("./Network.jl")
using DelimitedFiles
N = 1000000
data = zeros(50,N)
ms = [2,4,8,16,32]

for m in 1:5
    for n in 1:10
        println((m-1)*10+n)
        graph = BA_model(ms[m], N)

        data[(m-1)*10+n,:] = degree(graph)
    end
end
writedlm( "PA_data_2.csv",  data, ',')



