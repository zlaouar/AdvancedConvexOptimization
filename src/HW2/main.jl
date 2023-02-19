using AdvancedConvexOptimization
using Distributions
using Convex, ECOS
using Plots
using Optim
using ProximalAlgorithms
using ProximalCore
using LinearAlgebra
const aco = AdvancedConvexOptimization

# Problem 1 _____________________________________
N = 100
x = zeros(N)
x[10] = 1
x[13] = -1
x[50] = 0.3
x[70] = -0.2
y = aco.conv(x,hfunc)


B = implicit2explicit(blur, N)

# get blurred and noisy signal
σ = 0.02
dist = Normal(0,σ^2)
z = rand(dist, N)
y = B*x + z

# Problem 2 - solve the model _____________________________________
ϵ = σ*sqrt(N)
x1 = Convex.Variable(N)
constraint = square(norm(B*x1-y, 2)) <= ϵ^2
p = minimize(norm(x1,1), constraint)
solve!(p, ECOS.Optimizer, silent_solver = true)

plt=plot(x, title="Signal Reconstruction", label="Original")
plot!(y, label="Blurred and Noisy")
plot!(x1.value, label="Deblurred")
savefig(plt, "blur.png")

# Problem 3 - scale up _____________________________________
λ = constraint.dual
x2 = Convex.Variable(N)
p1 = minimize(norm(x2,1) + λ*square(norm(B*x2-y, 2)))
solve!(p1, ECOS.Optimizer, silent_solver = true)



# Problem 4 _____________________________________
βval = β(x)
adjβval = adjβ(x)

B1 = implicit2explicit(β, N)
B_ = implicit2explicit(adjβ, N)

isapprox(B_,adjoint(B1), atol=1e-8)

# Problem 5 _____________________________________
τ = 1/(2*λ)
f(x) = τ*norm(x,1) + 0.5*norm(B1*x-y, 2).^2
x0 = zeros(N)

# FIRST ORDER SOLVER NOT WORKING

res = optimize(f, x0, GradientDescent())

(b, b_ls) = LassoEN(y, B, τ)

println("Lasso coeffs (with γ=$τ)")
display(["Lasso"; b])

function ProximalCore.gradient(f,x)
    return similar(x), adjβ(β(x)-y)
end

ffb = ProximalAlgorithms.FastForwardBackward()
L = opnorm(B1,2)^2
gfunc2(x) = τ*norm(x,1)
gfunc1 = ProximalCore.ConvexConjugate(gfunc2)

solution, iterations = ffb(x0=x0, f=x->0.5*norm(β(x)-y, 2).^2, g=gfunc1, Lf=L)
