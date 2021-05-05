# developed with Julia 1.5.3
#
# tests for ControlVariables package

using ControlVariables, Test
const CV = ControlVariables

current_directory = @__DIR__


@testset "ControlVariables" begin
    
    function test_bounds()
        bounds = CV.Bounds(2, [10., 2.], [7., -1.])
        upper, _ = bounds[2]
        return upper == [10., 2.]
    end

    function test_scalar_bounds()
        bounds = CV.Bounds(2, 2., 1.)
        upper, _ = bounds[2]
        return upper == [2.]
    end

    function test_states_1()
        states = States(2, 1:5., 1:5.)
        upper, lower = states.bounds[2]
        return (upper == [5., 5.] && lower == [1., 1.])
    end

    function test_states_2()
        states = States(2, 1:5., 1:5.)
        x = [state for (state, index) in states.iterator]
        return x[end] == (5., 5.)
    end

    function test_controls()
        controls = Controls(2, 1:2., 1:2.)
        x = [i for i in controls[2]]
        return x[1] == (1., 1.)
    end

    function test_control_bounds()
        bounds = CV.Bounds(2, [10., 2.], [7., -1.])
        controls = Controls(bounds, 1:7., 1:5.)
        x = [i for i in controls[2]]
        return x[end] == (7., 2.)
    end

    function test_non_zero_random_variable()
        value = [1., 2., 3.]
        probability = [0. , 0.5, 0.5]
        rv = CV.RandomVariable(value, probability)
        return rv.probability == [0.5 , 0.5]
    end

    function test_noise_iterator_1d()
        w = collect(reshape(collect(1:6.0), 3, 2)')
        pw = ones(2, 3)*0.5

        noises = Noises(w, pw)
        noise = CV.RandomVariable(noises, 3)
        
        for (val, proba) in CV.law(noise)
            if (val[1], proba) == (6.0, 0.5)
                return true
            end
        end
        return false
    end

    function test_noise_iterator_2d()
        w = reshape(collect(1:12.0), 2, 2, 3)
        pw = ones(2, 3)*0.5
        noises = Noises(w, pw)
        noise = CV.RandomVariable(noises, 3)
        for (val, proba) in CV.law(noise) 
            if (val, proba) == ([11.0, 12.0], 0.5)
                return true
            end
        end
        return false
    end

    function test_noise_kmeans()
        data = rand(100, 5)
        noise = Noises(data, 3)
        for t in 1:5
            s = sum(noise[t].probability)
            if !isapprox(s, 1.0)
                return false
            end
        end                
        return true
    end

    function test_linear_noise_model()
        data = hcat(ones(5), ones(5)*2)
        w, n = ControlVariables.fit_linear_noise_model(data, 2)
        return all(isapprox.(w, [0.4 1.0 ; 0.2  1.0]))
    end

    @test test_bounds()
    @test test_states_1()
    @test test_states_2()
    @test test_scalar_bounds()
    @test test_controls()
    @test test_control_bounds()
    @test test_non_zero_random_variable()
    @test test_noise_iterator_1d()
    @test test_noise_iterator_2d()
    @test test_noise_kmeans()
    @test test_linear_noise_model()
    
end