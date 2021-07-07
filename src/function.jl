## developed with julia 1.5.3
#
## functions for optimal control variables


# Bounds constructors

function Bounds(horizon::Int64, upper::Array{Float64,1}, lower::Array{Float64,1})
	return Bounds(fill(upper, horizon), fill(lower, horizon))
end

function Bounds(horizon::Int64, upper::Float64, lower::Float64)
	return Bounds(fill([upper], horizon), fill([lower], horizon))
end

# States constructor

function States(horizon::Int64, axis::Vararg{T}) where T <: StepRangeLen{Float64}

	axis = collect(axis)
	upper = Float64[]
	lower = Float64[]
	indices = []

	for i in 1:length(axis)
		x = axis[i]
		x_min, x_max = extrema(x)
		push!(upper, x_max)
		push!(lower, x_min)
		push!(indices, 1:length(x))
	end

	indices = Iterators.product(indices...)
	iterator = zip(Iterators.product(axis...), indices)
	bounds = Bounds(horizon+1, upper, lower)

	return States(axis, iterator, bounds)

end

# Controls constructors

function Controls(horizon::Int64, axis::Vararg{T}) where T <: StepRangeLen{Float64}
	return Controls(fill(Iterators.product(axis...), horizon))
end

function Controls(bounds::Bounds, axis::Vararg{T}) where T <: StepRangeLen{Float64} 

	dimension = length(axis)
	iterators = Iterators.ProductIterator[]

	for t in 1:length(bounds.upper)

		upper, lower = bounds[t]
		axis_iterators = []

		for i in 1:dimension
			in_bounds(control::Float64) = lower[i] <= control <= upper[i]
			push!(axis_iterators, Iterators.filter(in_bounds, axis[i]))
		end

		push!(iterators, Iterators.product(axis_iterators...))

	end

	return Controls(iterators)

end

# Noises constructors

function Noises(value::Array{Float64,3}, probability::Array{Float64,2})

	horizon = size(value, 3)
	if horizon != size(probability, 2)
		error("horizon mismatch - values : $(horizon) != probabilities : $(size(probability, 2))")
	end

	variables = Array{RandomVariable,1}(undef, horizon)

	for t in 1:horizon
		rv = RandomVariable(value[:, :, t], probability[:, t])
		variables[t] = rv
	end

	return Noises(variables)

end

function Noises(value::Array{Float64,2}, probability::Array{Float64,2})
	cardinal, horizon = size(value)
	reshaped_value = reshape(value, 1, cardinal, horizon)
	return Noises(reshaped_value, probability)
end

function Noises(data::Array{Float64,2}, k::Int64) 

	"""dicretize noise space to k values using Kmeans: return type Noises
	data > time series data of dimension (n_data, horizon)
	k > Kmeans parameter

	"""

	n_data, horizon = size(data)
	w = zeros(k, horizon)
	pw = zeros(k, horizon)

	for t in 1:horizon
		w_t = reshape(data[:, t], (1, :))
		kmeans_w = kmeans(w_t, k)
		w[:, t] = kmeans_w.centers
		pw[:, t] = kmeans_w.counts / n_data
	end

	return Noises(w, pw)

end

# RandomVariable

RandomVariable(n::Noises, t::Int64) = n[t]
function RandomVariable(value::Array{Float64,1}, probability::Array{Float64,1})
	return RandomVariable(reshape(value, 1, length(value)), probability)
end
law(rv::RandomVariable) = zip(eachcol(rv.value), Iterators.Stateful(rv.probability))

# Variables constructor 

Variables(t::Int64, rv::RandomVariable) = Variables(t, nothing, nothing, rv)