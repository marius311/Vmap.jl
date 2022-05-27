module Vmap

using Base.Broadcast
using Zygote

export batch, Batched, vmap

# array wrapper indicating last dimension of array is batch dimension
struct Batched{N,T,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    x :: A
end
Zygote.@adjoint Batched(x) = Batched(x), Δ -> (Δ.x,)
batch(v::Vector{<:AbstractVector{T}}) where {T,N} = Batched(reduce(hcat, v))

# minimal AbstractArray interface for printing
Base.size(b::Batched) = Base.size(b.x)
Base.@propagate_inbounds Base.getindex(b::Batched, I::Union{Int,Colon,AbstractArray}...) = Base.getindex(b.x, I...)
Base.@propagate_inbounds Base.setindex!(b::Batched, X, I::Union{Int,Colon,AbstractArray}...) = (Base.setindex!(b.x, X, I...); b)

# preserve Batched wrapper when broadcasting
struct BatchedStyle{N,S<:Broadcast.AbstractArrayStyle{N}} <: Broadcast.AbstractArrayStyle{N} end
Broadcast.BroadcastStyle(::Type{Batched{N,T,A}}) where {N,T,A} = BatchedStyle{N,typeof(BroadcastStyle(A))}()
Broadcast.BroadcastStyle(::BatchedStyle{N,S}, ::BatchedStyle{N,S}) where {N,S} = BatchedStyle{N,S}()
Broadcast.BroadcastStyle(::BatchedStyle{N,S}, ::Broadcast.DefaultArrayStyle) where {N,S} = BatchedStyle{N,S}()
Broadcast.preprocess(dest_tyle::BatchedStyle{N,S}, bc::Broadcast.Broadcasted) where {N,S} =
    Broadcast.broadcasted(S(), bc.f, Broadcast.preprocess_args(dest_tyle, bc.args)...)
Broadcast.preprocess(dest_style, b::Batched) = b.x
function Broadcast.materialize(bc::Broadcast.Broadcasted{BatchedStyle{N,S}}) where {N,S}
    bc′ = Broadcast.preprocess(BatchedStyle{N,S}(), bc)
    bc″ = convert(Broadcast.Broadcasted{S}, bc′)
    Batched(Broadcast.materialize(bc″))
end

# batch-aware reductions
function Base.sum(b::Batched{N}) where {N}
    dims = ntuple(identity, N-1)
    dropdims(sum(b.x; dims); dims)
end

vmap(f) = x -> f(batch(x))

end
