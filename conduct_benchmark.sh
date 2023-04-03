#!/bin/bash

mesa_program="model.py"
julia_program="model.jl"

declare -a agents=(1 2 4 8 16 32 64 128 256)
declare -a threads=(1 2 4 7)

for agent in "${agents[@]}"; do
    # Run the Mesa program
    python ./"$mesa_program" "$agent"

    # Run the Julia program with different numbers of threads
    for thread in "${threads[@]}"; do
        JULIA_NUM_THREADS="$thread" julia "$julia_program" "$agent"
    done
done
