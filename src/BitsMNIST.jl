module BitsMNIST
	
	using Reexport
	@reexport using Flux
	@reexport using TinyML

	include("./Modules/Misc.jl")
	using .Misc
	include("./Modules/Datasets.jl")
	using .Datasets
	include("./Modules/IO.jl")
	using .IO
	include("./Modules/Statistics.jl")
	using .Statistics
	include("./Modules/ZeroOne.jl")
	using .ZeroOne

end