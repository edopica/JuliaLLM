"""
bootstrap.jl — install and precompile all project dependencies.

Run once after cloning:
    julia scripts/bootstrap.jl
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
println("Dependencies installed and precompiled.")
