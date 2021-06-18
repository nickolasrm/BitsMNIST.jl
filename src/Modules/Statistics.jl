module Statistics

	using Flux

	"""
		error(model::Chain, x::Vector, y::Vector)

	Calculates the percentage of errored predictions.
	"""
	function error(model::Chain, x::Vector, y::Vector)
		errored = 0
		for (xi, yi) in zip(x, y)
			if yi != (argmax(model(xi))-1)
				errored += 1
			end
		end
		errored / length(y)
	end

	export error

end