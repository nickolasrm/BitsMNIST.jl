@testset "Datasets" begin
	
	mnist = BitsMNIST.Datasets.mnist()
	@test length(mnist["train_x"]) == 60000
	@test length(mnist["train_x"][1]) == 784
	@test length(mnist["test_x"][1]) == 784
	@test length(mnist["train_y"]) == 60000

	noisy = BitsMNIST.Datasets.noisymnist()
	@test noisy["test_y"][1] == BitsMNIST.Datasets.NOISE_LABEL
	
end