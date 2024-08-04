using Graphs
using StatsBase
using ProgressMeter

#return vertices for new node to be added to at random in preferential attachment.
function BA_time_step(g,edges, m)
    counter = 0 
    add_vertex!(g)
    neighbours = []
    while counter != m#required in case a node is picked twice from the list of edges
        potential_neighbour = rand((edges))
        if potential_neighbour ∉ neighbours
            add_edge!(g,nv(g), potential_neighbour)
            counter +=1 
            append!(neighbours, potential_neighbour)
        end
    end

    for vertex in neighbours
        append!(edges,[nv(g), vertex])
    end
end

function rand_time_step(g,size, m)
    counter = 0 
    add_vertex!(g)
    neighbours = []
    while counter != m#required in case a node is picked twice from the list of edges
        potential_neighbour = rand(1:size)
        if potential_neighbour ∉ neighbours
            add_edge!(g,nv(g), potential_neighbour)
            counter +=1 
            append!(neighbours, potential_neighbour)
        end
    end

    size+=m
end

#regular graph with m nodes and m-1 edges each by default.
function init_graph_reg(m, k = m-1)
    return random_regular_graph(m, k)
end


function BA_model(m, N)

   Graph = init_graph_reg(m+1)
   edge = []
   for i in edges(Graph)

       append!(edge, [src(i), dst(i)])
   end
#progress bar so I can see how speedy my julia code is 
@showprogress 1 "Computing..."  for i in 1:(N-m-1)
    BA_time_step(Graph, edge,m)
    end

   return Graph
end

function rand_attach(m, N)
    Graph = init_graph_reg(m+1)
    size = m+1
@showprogress 1 "Computing..."  for i in 1:(N-m-1)
    rand_time_step(Graph,size,m)
    size += 1
    end
    return Graph
end

function mixed_time_step(g, size, edges, m, r)
   #choosing existing vertices 
   sources = []
   ends = []
   for i in 1:(m-r)
        node1 = rand((edges))
        node2 = rand((edges))
        while node1 == node2
            node2 = rand((edges))
        end
        append!(sources, node1)
        append!(ends, node2)
    end   
    #choosing new vertex edges
    counter = 0 
    add_vertex!(g)
    neighbours = []
    while counter != r#required in case a node is picked twice from the list of edges
        potential_neighbour = rand(1:size)
        if potential_neighbour ∉ neighbours
            add_edge!(g,nv(g), potential_neighbour)
            counter +=1 
            append!(neighbours, potential_neighbour)

        end
    end
    #actually impelmenting the changes
    for i in 1:(m-r)
        add_edge!(g,sources[i],ends[i])
        append!(edges,sources[i],ends[i])
    end
    for i in 1:r
        append!(edges,nv(g),neighbours[i])
    end
end



function mixed(m, N)
    r = m÷3
    Graph = init_graph_reg(m+1)
    edge = []
    for i in edges(Graph)
 
        append!(edge, [src(i), dst(i)])
    end
    size = m+1
@showprogress 1 "Computing..."  for i in 1:(N-m-1)
    mixed_time_step(Graph,size,edge,m,r)
    size += 1
    end
    return Graph
end






