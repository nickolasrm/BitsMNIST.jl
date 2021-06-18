module NoisyMNIST

    using Pkg
    using MLDatasets
    using JLD2
    using ..Misc
    using ..MNIST

    const FILEPATH = joinpath(Misc.DATASETS_FOLDER, "noisybitsmnist.jld2")
    const NOISE_RATE = 0.3
    const NOISE_LABEL = -1

    function addnoise(set)
        for line in set
            for i in eachindex(line)
                if rand() < NOISE_RATE
                    line[i] != rand(Bool)
                end
            end
        end
    end

    """
        download()

    Forces downloading the neisy bits mnist dataset
    """
    function download()
        set = mnist()
        addnoise(set["train_x"])
        addnoise(set["test_x"])
        fill!(set["train_y"], NOISE_LABEL)
        fill!(set["test_y"], NOISE_LABEL)
        save(FILEPATH, set)
        set
    end

    """
        noisymnist()

    Downloads and return the noisy bits mnist dataset
    """
    function noisymnist()
        if isfile(FILEPATH)
            return load(FILEPATH)
        else
            return download()
        end
    end

    export noisymnist, NOISE_LABEL

end