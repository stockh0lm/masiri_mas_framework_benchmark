using Agents, Random, Distributions, CSV, BenchmarkTools, Base.Threads
using Printf: @printf

const FIB_NUM = 0
const RECORD_FILE = "./benchmark_results_julia.csv"

mutable struct User <: AbstractAgent
    id::Int
    trip_duration::Float64
    fibonacci_num::Int
    energy_used::Float64
    energy_generated::Float64
end

mutable struct House <: AbstractAgent
    id::Int
    size::Float64
    solar_power::Bool
    heat_pump::Bool
    air_conditioning::Bool
    electric_car::Bool
    fibonacci_num::Int
    energy_used::Float64
    energy_generated::Float64
end

mutable struct EnergyModel 
    agents::Vector{AbstractAgent}
    step::Int
    total_energy_used::Float64
    total_energy_generated::Float64
    temperature_data::Array{Float64, 2}
    sunshine_data::Array{Float64, 2}
    car_on_trip::Dict{Int, Bool}
end


#https://discourse.julialang.org/t/how-to-track-total-memory-usage-of-julia-process-over-time/91167/6

function fibonacci(n::Int)
    if n <= 1
        return n
    else
        return fibonacci(n - 1) + fibonacci(n - 2)
    end
end

function append_numbers_to_file(filename, numbers)
    open(filename, "a") do io
        write(io, join(numbers, ',') * "\n")
    end
end

function meminfo_procfs(pid=getpid())
    smaps = "/proc/$pid/smaps_rollup"
    if !isfile(smaps)
        error("`$smaps` not found. Maybe you are using an OS without procfs support or with an old kernel.")
    end

    rss = pss = shared = private = 0
    for line in eachline(smaps)
        s = split(line)
        if s[1] == "Rss:"
        rss += parse(Int64, s[2])
        elseif s[1] == "Pss:"
        pss += parse(Int64, s[2])
        elseif s[1] == "Shared_Clean:" || s[1] == "Shared_Dirty:"
        shared += parse(Int64, s[2])
        elseif s[1] == "Private_Clean:" || s[1] == "Private_Dirty:"
        private += parse(Int64, s[2])
        end
    end

    @printf "RSS:       %9.3f MiB\n" rss/2^10
    @printf "┝ shared:  %9.3f MiB\n" shared/2^10
    @printf "┕ private: %9.3f MiB\n" private/2^10
    @printf "PSS:       %9.3f MiB\n" pss/2^10
    return rss/2^10
end

function is_active(time_of_day::Float64)
    mean_active_time = 14  # Mean of the normal distribution at 14:00 (14 hours)
    std_dev = 3  # Standard deviation of 3 hours
    probability = pdf(Normal(mean_active_time, std_dev), time_of_day)
    return rand() < probability
end

function calculate_charging_time(trip_duration::Float64, battery_capacity::Float64, charging_power::Float64)
    energy_used = battery_capacity * trip_duration  # Assuming linear relationship between trip duration and energy used
    charging_time = energy_used / charging_power
    return charging_time
end

function user_step!(agent::User, model)
    time_of_day = (model.tick - 0.5) * 5 / 60  # Convert the current step to the hour of the day
    agent.fibonacci_num += fibonacci(FIB_NUM)
    if is_active(time_of_day)
        trip_duration = rand(model.rng, 0.5:0.5:2)  # Generate a random trip duration between 0.5 and 2 hours
        agent.trip_duration = trip_duration
        battery_capacity = Float64(60)  # Assume a 60 kWh battery capacity
        charging_power = 50 / 60  # Assume a 50 kW charging power (in kWh per 5-minute interval)
        charging_time = calculate_charging_time(trip_duration, battery_capacity, charging_power)
        
        agent.energy_used += trip_duration * battery_capacity / charging_time  # Add energy used to the total energy used
        agent.energy_generated += charging_time * charging_power  # Add energy generated (charging) to the total energy generated
    end
end

function house_properties()
    size = rand(Uniform(0.5, 2))  # Randomize house size between 0.5 and 2 (representing small to big houses)
    solar_power = rand() < 0.5  # 50% of houses have solar power
    heat_pump = rand() < 0.5  # 50% of houses have a heat pump
    air_conditioning = rand() < 0.5  # 50% of houses have air conditioning
    electric_car = rand() < 0.5  # 50% of houses have an electric car
    fibonacci_num = 0
    energy_used = 0.0
    energy_generated = 0.0
    return size, solar_power, heat_pump, air_conditioning, electric_car, fibonacci_num, energy_used, energy_generated
end

function energy_generation(house::House, sunshine_intensity::Float64)
    solar_power_capacity = 5 * house.size  # Bigger houses have more solar power capacity
    return house.solar_power ? sunshine_intensity * solar_power_capacity : 0
end

function energy_consumption(house::House, temperature::Float64)
    heat_pump_consumption = house.heat_pump ? max(0, 18 - temperature) * house.size : 0
    air_conditioning_consumption = house.air_conditioning ? max(0, temperature - 24) * house.size : 0
    return heat_pump_consumption + air_conditioning_consumption
end



function house_step!(agent::House, model)
    agent.fibonacci_num += fibonacci(FIB_NUM)

    n_days = 365
    intervals_per_day = convert(Int, 24 * 60 / 5)  # 5-minute intervals

    day_of_year = (ceil(Int, model.tick/intervals_per_day))
    time_of_day = model.tick%intervals_per_day+1

    
    temperature = model.properties[:temperature_data][day_of_year, time_of_day]
    sunshine_intensity = model.properties[:sunshine_data][day_of_year, time_of_day]
    
    generated_energy = energy_generation(agent, sunshine_intensity)
    consumed_energy = energy_consumption(agent, temperature)
    
    if agent.electric_car
        if model.properties[:car_on_trip][agent.id]
            car_energy = 0  # The car is on a trip and not available for energy storage or consumption
        else
            battery_capacity = 60  # Assume a 60 kWh battery capacity
            charging_power = 50 / 60  # Assume a 50 kW charging power (in kWh per 5-minute interval)
            car_energy = min(generated_energy - consumed_energy, charging_power)  # Car energy is limited by the charging power
            car_energy = max(car_energy, -battery_capacity)  # Car energy cannot be negative beyond the battery capacity
        end
    else
        car_energy = 0
    end
    
    net_energy = generated_energy - consumed_energy + car_energy
    agent.energy_generated += max(0, net_energy)
    agent.energy_used += max(0, -net_energy)
end

function luebeck_temperature(day_of_year::Int, time_of_day::Float64)
    daily_avg_temp = 2 * sin(2π * (day_of_year - 31) / 365) + 10  # Average temperature peaks in July
    daily_temp_variation = 5 * sin(2π * time_of_day / 24)  # Temperature peaks at 3 pm
    noise = rand(Normal(0, 1))  # Add realistic noise
    return daily_avg_temp + daily_temp_variation + noise
end

function luebeck_sunshine(day_of_year::Int, time_of_day::Float64)
    max_daylight_hours = 17  # Maximum daylight hours in Lübeck
    min_daylight_hours = 7  # Minimum daylight hours in Lübeck
    daylight_hours = min_daylight_hours + (max_daylight_hours - min_daylight_hours) / 2 * (1 + sin(2π * (day_of_year - 172) / 365))  # Daylight hours vary sinusoidally over the year
    sunrise = 12 - daylight_hours / 2
    sunset = 12 + daylight_hours / 2
    if sunrise <= time_of_day <= sunset
        sunshine_intensity = sin(π * (time_of_day - sunrise) / daylight_hours)  # Sunshine intensity peaks at solar noon
        noise = rand(Uniform(-0.1, 0.1))  # Add realistic noise
        return sunshine_intensity + noise
    else
        return 0
    end
end

function generate_temperature_and_sunshine_data()
    n_days = 365
    intervals_per_day = convert(Int, 24 * 60 / 5)  # 5-minute intervals
    temperature_data = Array{Float64, 2}(undef, n_days, intervals_per_day)
    sunshine_data = Array{Float64, 2}(undef, n_days, intervals_per_day)

    for day_of_year in 1:n_days
        for interval in 1:intervals_per_day
            time_of_day = (interval - 0.5) * 5 / 60  # Convert the interval to the hour of the day
            temperature_data[day_of_year, interval] = luebeck_temperature(day_of_year, time_of_day)
            sunshine_data[day_of_year, interval] = luebeck_sunshine(day_of_year, time_of_day)
        end
    end

    return temperature_data, sunshine_data
end


function create_energy_model(n_agents::Int64)
    n_days = 365
    intervals_per_day = convert(Int, 24 * 60 / 5)  # 5-minute intervals
    temperature_data, sunshine_data = generate_temperature_and_sunshine_data()

    n_users = n_agents
    n_houses = n_agents

    users = [User(i, 0.0, 0,0.0,0.0) for i in 1:n_users]
    houses = [House(i + n_users, house_properties()...) for i in 1:n_houses]

    agents = vcat(users, houses)
    car_on_trip = Dict(i => false for i in (n_users+1):(n_users+n_houses))

    properties = Dict(
        :total_energy_used => 0.0,
        :total_energy_generated => 0.0,
        :temperature_data => temperature_data,
        :sunshine_data => sunshine_data,
        :car_on_trip => car_on_trip,
        :tick => 1
    )

    model = ABM(Union{User, House}; properties=properties, warn=false)
    for agent in agents
        add_agent!(agent, model)
    end

    return model
end

function agent_step!(agent, model)
    if isa(agent, User)
        user_step!(agent, model)
    elseif isa(agent, House)
        house_step!(agent, model)
    end
end

function model_step!(model)
    model.tick += 1
    @threads for agent in collect(allagents(model))
        agent_step!(agent, model)
    end
    for agent in allagents(model)
        model.properties[:total_energy_used] += agent.energy_used
        model.properties[:total_energy_generated] += agent.energy_generated
    end
    model.properties[:total_energy_used] = sum(model[id].energy_used for id in allids(model))
    model.properties[:total_energy_generated] = sum(model[id].energy_generated for id in allids(model))
end



function main(args)
    num_agents::Int64 = 0
    if length(args) != 1
        println("please use with one numeric argument for number of agents")
        return
    end
    num_agents = parse(Int, args[1])

   avg_time = @belapsed begin
        model = create_energy_model($num_agents)
        n_days = 365
        intervals_per_day = convert(Int, 24 * 60 / 5)  # 5-minute intervals
        n_steps = n_days * intervals_per_day
        _, mdata = run!(model, dummystep, model_step!, n_steps-1, mdata = [:total_energy_used,:total_energy_generated])
        #pid::Int = getpid() 
        #filename = "mdata-$(pid).csv"
        #CSV.write(filename, mdata)
    #    println("Total energy used: ", model.total_energy_used, "\tTotal energy generated: ", model.total_energy_generated)
    end    
    rss = meminfo_procfs()
    append_numbers_to_file(RECORD_FILE, [num_agents, nthreads(), avg_time, rss])
end

main(ARGS)