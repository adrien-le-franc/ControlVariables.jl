## developed with julia 1.5.3
#
## noise models for optimal control variables


function fit_linear_noise_model(data::Array{Float64,2}, k::Int64=10)

	n_data, horizon = size(data)
	weights = zeros(2, horizon)
	support = zeros(k, horizon)
	probability = zeros(k, horizon)

	for t in 1:horizon

		if t == 1
			x = data[:, horizon]
			y = data[:, t]
		else
			x = data[:, t-1]
			y = data[:, t]
		end

		x = hcat(x, ones(size(x)))
		weights[:, t] = pinv(x'*x)*x'*y
		epsilon = y - x*weights[:, t]

		epsilon = reshape(epsilon, (1, :))
		kmeans_w = kmeans(epsilon, k)
		support[:, t] = kmeans_w.centers
		probability[:, t] = kmeans_w.counts / n_data 

	end

	return weights, Noises(support, probability)

end