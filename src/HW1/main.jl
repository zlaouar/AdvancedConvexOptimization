using Convex 
using ECOS
using StatsBase 

using CSV, DataFrames

# read in data
data = CSV.read("HW1/winequality-white.csv", DataFrame)
X = Matrix(data)[:,1:end-1]
y = data[:,end]

n = ncol(data)-1
β = Convex.Variable(n)

problem1 = minimize(norm(y - X*β, 2))
solve!(problem1, ECOS.Optimizer, silent_solver = true)
println(problem1.optval)
println(β.value)
l2val = β.value
β1 = deepcopy(β)

# L1 norm
problem2 = minimize(norm(y - X*β, 1))
solve!(problem2, ECOS.Optimizer, silent_solver = true)
println(problem2.optval)
l1val = β.value
β2 = deepcopy(β)

zscores = zscore(y)
σ = std(y)
oldy = deepcopy(y)
inds = []
for (i,score) in enumerate(zscores)
    if abs(score) > 3*σ
        push!(inds, i)
    end
end
inds = setdiff(collect(1:length(y)),inds)
Xnew = X[inds, :]
ynew = y[inds]
plot(y, title="White Wine Quality with and without Outliers", label="No Outliers")
plot!(ynew, label="With Outliers")


# Outliers optimization
problem3 = minimize(norm(ynew - Xnew*β, 2))
solve!(problem3, ECOS.Optimizer, silent_solver = true)
println(problem3.optval)
println(β.value)
l2valOut = β.value
β3 = deepcopy(β)

# L1 norm
problem4 = minimize(norm(ynew - Xnew*β, 1))
solve!(problem4, ECOS.Optimizer, silent_solver = true)
println(problem4.optval)
l1valOut = β.value
β4 = deepcopy(β)


# Residuals
xrange = collect(-4:0.5:4)
res1 = y-X*β1.value
res2 = y-X*β2.value
res3 = ynew-Xnew*β3.value
res4 = ynew-Xnew*β4.value

ϕ1(val) = abs(val)
ϕ2(res) = res^2
plot(xrange,ϕ1.(xrange))
histogram(res1, fillalpha=0.9, labels="l1-norm", title="Quality Residuals - with Outliers")
histogram!(res2, fillalpha=0.9, xlims=(-4,4), labels="l2-norm")
plot!(xrange, 110*ϕ1.(xrange), linewidth=2, label="l1-penalty")
display(plot!(xrange, 30*ϕ2.(xrange), linewidth=2, label="l2-penalty"))

histogram(res3, fillalpha=0.9, labels="l1-norm", title="Quality Residuals - without Outliers")
histogram!(res4, fillalpha=0.9, xlims=(-4,4), labels="l2-norm")
plot!(xrange, 110*ϕ1.(xrange), linewidth=2, label="l1-penalty")
plot!(xrange, 30*ϕ2.(xrange), linewidth=2, label="l2-penalty")