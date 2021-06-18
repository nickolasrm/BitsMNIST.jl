module MNIST

    using Pkg
    using MLDatasets
    using JLD2
    using ..Misc

    const FILEPATH = joinpath(Misc.DATASETS_FOLDER, "bitsmnist.jld2")

    function avgwhite(x)
        s = sum(x)
        cnt = count(x -> x > 0, x)
        s / cnt
    end

    function convert2binary(x)
        avg = avgwhite(x)
        rx = reshape(x, Misc.IMG_LENGTH, size(x, 3))
        [BitVector(rx[:, i] .> avg) for i in axes(x, 3)]
    end

    """
        download()

    Forces downloading the bits mnist dataset
    """
    function download()
        MLDatasets.MNIST.download(i_accept_the_terms_of_use=true)
        train_x, train_y = MLDatasets.MNIST.traindata()
        test_x, test_y = MLDatasets.MNIST.testdata()

        btrain_x, btest_x = convert2binary(train_x), convert2binary(test_x)
        set = Dict("train_x" => btrain_x, 
                                    "train_y" => train_y, 
                                    "test_x" => btest_x, 
                                    "test_y" => test_y)
        save(FILEPATH, set)
        set
    end

    """
        mnist()

    Downloads and return the bits mnist dataset
    """
    function mnist()
        if isfile(FILEPATH)
            return load(FILEPATH)
        else
            return download()
        end
    end

    export mnist

end