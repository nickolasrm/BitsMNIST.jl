"""
    sample(x::Vector, y::Vector, fraction::Real)

Returns a sample of 0 and 1 examples of the dataset balanced in a 50/50 proportion with
the length of length(x)*fraction.

Example: If you provide 60000 examples with a 0.01 fraction, a sample with 600 examples will
be returned.
"""
function sample(x::Vector, y::Vector, fraction::Real)
    sample_size = ceil(Int, fraction * length(y))
    sx = Vector{BitVector}(undef, sample_size)
    sy = Vector{Int}(undef, sample_size)
    
    eachmax = sample_size / 2
    zeros = 0
    ones = 0
    counter = 0

    function extractentry(i::Int) 
        counter += 1
        sx[counter] = x[i]
        sy[counter] = y[i]
    end

    for i in eachindex(y)
        if counter == sample_size 
            break 
        elseif y[i] == 1
            if ones < eachmax
                ones += 1
                extractentry(i)
            end
        elseif y[i] == 0
            if zeros < eachmax
                zeros += 1
                extractentry(i)
            end
        end
    end

    if counter < sample_size
        sx = sx[1:counter]
        sy = sy[1:counter]
    end

    (sx, sy)
end