@testset "IO" begin
	
	Random.seed!(SEED)

	#Dataset
	dset = BitsMNIST.Datasets.mnist()
	sx, sy = BitsMNIST.ZeroOne.sample(dset["train_x"], dset["train_y"], 0.01) #600 examples

	#Model
	model = Chain(BitDense(784, 800), BitDense(800, 2, true, sigmoid))

	#Genetic
	fitness = BitsMNIST.ZeroOne.generate_score_fitness(sx, sy)
	tset = Genetic.TrainingSet(model, model.layers, fitness, mutationRate=0.05)
	Genetic.train!(tset, genNumber=1)

	BitsMNIST.IO.save("./test.jld2", model, tset)
	@test isfile("./test.jld2")

	ltset = BitsMNIST.IO.load("./test.jld2")
	display(ltset)
	@test sum(ltset["model"](sx[1]) .== model(sx[1])) == 2

end