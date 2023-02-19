function hfunc(j)
    return exp((-(j-3)^2)/2)
end

function wrap(j,i,N)
    ind = j-i+1
    if ind < 1
        return N + ind
    elseif ind > N
        return ind - N
    end
    return ind
end

function conv(x,hfunc)
    h = [hfunc(j) for j in 1:5]
    L = length(h)
    lx = length(x)
    y = zeros(lx)
    for j in 1:lx
        y[j] = sum([x[wrap(j,i,lx)]*h[i] for i in 1:L])
    end
    return y
end

function blur(x)
    conv(x,hfunc)
end

function implicit2explicit(Bx::Function, N)
    mat = diagm(N,N,ones(N))
    B = Array{eltype(Bx(ones(N)))}(undef, N, N)
    for i in 1:N
        B[:,i] = Bx(mat[:,i])
    end
    return B
end

function Bx(x)
    mat = diagm(N,N,ones(N))
    return mat*x
end

function adjB(h,z)
    ĥ = conj(h)
    return ĥ .* z
end

function β(x)
    N = length(x)
    h = [hfunc(j) for j in 1:5]
    ĥ = [fft(h); zeros(N-length(h))]
    Fx = fft(x)
    H = ĥ .* Fx
    return ifft(H)
end

function adjβ(x)
    N = length(x)
    h = [hfunc(j) for j in 1:5]
    ĥ = [fft(h); zeros(N-length(h))]
    adjB = fft(x)
    adjB = conj(ĥ) .* adjB
    return ifft(adjB)
end

function testAdjoint(β,β_,N)
    # check if β_ is adjoint of β
    implicit2explicit(β_, N)
end

