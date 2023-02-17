using AdvancedConvexOptimization
using Distributions
using Convex, ECOS
using Plots

N = 100
x = zeros(N)
x[10] = 1
x[13] = -1
x[50] = 0.3
x[70] = -0.2
#h = [hfunc(j) for j in 1:5]
y = conv(x,hfunc)


B = implicit2explicit(blur, N)

# get blurred and noisy signal
σ = 0.02
dist = Normal(0,σ^2)
y = B*x + rand(dist, N)

# solve the model
ϵ = σ*sqrt(N)
x1 = Convex.Variable(N)
constraint = norm(B*x1-y, 2) >= ϵ^2
p = minimize(norm(x,1), constraint)
solve!(p, ECOS.Optimizer, silent_solver = true)
println(p.optval)
println(x1.value)

plot(x, title="Signal Reconstruction", label="Original")
plot!(y, label="Blurred and Noisy")
plot!(x1.value, label="Deblurred")

# scale up
λ = Convex.Variable()
p1 = minimize(norm(x,1) + λ*norm(B*x1-y, 2), constraint)
solve!(p1, ECOS.Optimizer, silent_solver = true)
println(p1.optval)
println(x1.value)
