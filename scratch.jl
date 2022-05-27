
using Vmap, BenchmarkTools, Zygote

N_batch = 10
x = [rand(100000) for i = 1:N_batch]

relu(x) = x > 0 ? x : zero(x)
loss(x) = sum(relu.(x))
loss_batch(x) = dropdims(sum(relu.(x), dims=1), dims=1)

@btime vmap(loss)(x)
@btime loss_batch(reduce(hcat, x))

Zygote.gradient(x -> sum(vmap(loss)(x)), x)[1]