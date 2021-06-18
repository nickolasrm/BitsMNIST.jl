module Misc

	using Pkg

	const DATASETS_FOLDER = joinpath(dirname(dirname(@__DIR__)), "datasets")
	const IMG_LENGTH = 784

end