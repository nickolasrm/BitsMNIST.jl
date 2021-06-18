@testset "Statistics" begin
	
	set = BitsMNIST.Datasets.mnist()
	sx, sy = BitsMNIST.ZeroOne.sample(set["train_x"], set["train_y"], 0.01)
	model = Chain(BitDense(784, 2, true, sigmoid))
	error = BitsMNIST.Statistics.error(model, sx, sy)

	@test error >= 0
	@test error <= 1

end