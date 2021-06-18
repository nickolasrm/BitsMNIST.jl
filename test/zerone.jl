@testset "ZeroOne" begin

	@testset "Score Fitness" begin 
		Random.seed!(SEED)

		#Dataset
		dset = BitsMNIST.Datasets.mnist()
		sx, sy = BitsMNIST.ZeroOne.sample(dset["train_x"], dset["train_y"], 0.01) #600 examples

		#Model
		model = Chain(BitDense(784, 800), BitDense(800, 2))

		#Error t0
		error0 = BitsMNIST.Statistics.error(model, sx, sy)

		#Genetic: it will have to be updated in the next major release
		fitness = BitsMNIST.ZeroOne.generate_score_fitness(sx, sy)
		tset = Genetic.TrainingSet(model, model.layers, fitness, mutationRate=0.05)
		Genetic.train!(tset, genNumber=5)
		
		#Error t1
		error1 = BitsMNIST.Statistics.error(model, sx, sy)

		@test error1 < error0
	end

	@testset "MCC Fitness" begin 
		Random.seed!(SEED)

		#Dataset
		dset = BitsMNIST.Datasets.mnist()
		sx, sy = BitsMNIST.ZeroOne.sample(dset["train_x"], dset["train_y"], 0.01) #600 examples

		#Model
		model = Chain(BitDense(784, 800), BitDense(800, 2))

		#Error t0
		error0 = BitsMNIST.Statistics.error(model, sx, sy)

		#Genetic: it will have to be updated in the next major release
		fitness = BitsMNIST.ZeroOne.generate_mcc_fitness(sx, sy)
		tset = Genetic.TrainingSet(model, model.layers, fitness, mutationRate=0.05)
		Genetic.train!(tset, genNumber=5)
		
		#Error t1
		error1 = BitsMNIST.Statistics.error(model, sx, sy)

		@test error1 < error0
	end

end