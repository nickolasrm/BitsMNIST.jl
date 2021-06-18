module Datasets

	using Reexport
	using ..Misc

	include("./Datasets/MNIST.jl")
	@reexport using .MNIST
	include("./Datasets/NoisyMNIST.jl")
	@reexport using .NoisyMNIST

end