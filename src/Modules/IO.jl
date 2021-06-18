module IO

	using JLD2
	using Flux
	using TinyML

	"""
		load(::String)

	Loads a jld2 model from the informed current path
	"""
	load(filename::String) = JLD2.load(filename)

	"""
		save(model::Chain, training_set::Genetic.TrainingSet)

	Saves a jld2 model to the current path
	"""
	function save(filename::String, model::Chain, trainingset::Genetic.TrainingSet)
		tosave = Dict("model" => model,
					"trainingset" => trainingset)
		JLD2.save(filename, tosave)
	end	

	export load, save

end