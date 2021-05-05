## developed with julia 1.5.3
#
## struct for optimal control problems

# Type for bounds

struct Bounds
	upper::Array{Array{Float64,1},1}
	lower::Array{Array{Float64,1},1}

	function Bounds(upper::Array{Array{Float64,1},1}, lower::Array{Array{Float64,1},1})
		if length(upper) != length(lower)
			error("Bounds: length of upper $(length(upper)) != lower $(length(lower))")
		end
		for t in 1:length(upper)
			if !all(lower[t] .<= upper[t])
				error("Bounds: empty domain at step $(t), 
					where lower $(lower[t]) and upper $(upper[t])")
			end
		end
		new(upper, lower)
	end
end

Base.size(bounds::Bounds) = length(bounds.upper)
Base.getindex(bounds::Bounds, index::Int64) = bounds.upper[index], bounds.lower[index]

# Type for states

struct States{Ta <: StepRangeLen{Float64}, Ti <: Iterators.Zip}
	axis::Array{Ta}
	iterator::Ti
	bounds::Bounds
end

Base.size(states::States) = size(states.iterator)

# Type for controls

struct Controls{T <: Iterators.ProductIterator}
	iterators::Array{T,1}
end

Base.size(controls::Controls) = length(controls.iterators)
Base.getindex(controls::Controls, index::Int64) = controls.iterators[index]

# Type for random variables

struct RandomVariable
	value::Array{Float64,2}
	probability::Array{Float64,1}

	function RandomVariable(value::Array{Float64,2}, probability::Array{Float64,1})
		if size(value, 2) != length(probability)
			error("#support : $(size(value, 2)) != #probabilities : $(length(probability))")
		end
		if !isapprox(sum(probability), 1.)
			error("sum probability != 1.")
		end
		if 0. in probability
			n = sum(probability .!= 0.)
			new_value = zeros(size(value, 1), n)
			new_probability = zeros(n)
			non_zeros = 1
			for (i, p) in enumerate(probability) 
				if p != 0.
					new_value[:, non_zeros] = value[:, i]
					new_probability[non_zeros] = p
					non_zeros += 1
				end
			end
			probability = new_probability
			value = new_value
		end
		new(value, probability)
	end

end

Base.length(rv::RandomVariable) = length(rv.probability)

# Type for noises

struct Noises
	variables::Array{RandomVariable,1}
end

Base.length(n::Noises) = length(n.variables)
Base.getindex(n::Noises, t::Int64) = n.variables[t]

# Type containing all variables for Stochastic Optimal Control

mutable struct Variables
	t::Int64
	state::Union{Array{Float64,1}, Nothing}
	control::Union{Array{Float64,1}, Nothing}
	noise::Union{RandomVariable, Nothing}
end
