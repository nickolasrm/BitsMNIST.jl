module Reinforcement

	using MLJ: categorical, mcc
	using Flux

	"""
		generate_score_fitness(x::Vector, y::Vector)

	Generates a fitness function that increases the fitness score depending on whether the
	model finds the right answer.

	Example: y = 0, predicted_y = 0, model_output = model(example), then 
				score += model_output[predicted_y+1]
			#takes the value of the respective output node
			
			 y = 0, predicted_y = 1, model_output = model(example), then
			 	score = score
			#does nothing
	"""
	function generate_score_fitness(x::Vector, y::Vector)
		function fitness(chain::Chain)
			score = 0.0
			for (xi, yi) in zip(x, y)
				out = chain(xi)
				predicted = argmax(out) - 1
				if predicted == yi
					if yi == 1 
						score += out[2]
					else 
						score += out[1]
					end
				end
			end
			score
		end
	end

	"""
		generate_mcc_fitness(x::Vector, y::Vector)

	Generates a fitness function that increases the fitness score depending on the 
	Matthews Correlation Coefficient (MCC).
	"""
	function generate_mcc_fitness(x::Vector, y::Vector)
		predicted = categorical(zeros(Int, length(y)))
		caty = categorical(y)
		function fitness(chain::Chain)
			for (xi, i) in zip(x, eachindex(y))
				out = chain(xi)
				predicted[i] = argmax(out) - 1
			end
			mcc(caty, predicted)
		end
	end

	export generate_score_fitness, generate_mcc_fitness

end