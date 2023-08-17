
using StatsModels, MixedModels, DataFrames
import DSP.conv
using Unfold
using UnfoldMakie,CairoMakie
using DataFrames
using CategoricalArrays
using HDF5
using CSV


sbj_id = ARGS[1]

c = h5open("/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/EMEG_data_sbj_$sbj_id.h5", "r") do file
    read(file, "eeg")
end


data = transpose(c)

# evts = CSV.read("/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/evts_sbj_5_correct.csv", DataFrame)
evts = CSV.read("/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/evts_sbj_$(sbj_id)_concpred.csv", DataFrame)

bf1 = firbasis(τ=(-0.15,.5),sfreq=250,name="fix")
bf2 = firbasis(τ=(-0.15,.5),sfreq=250,name="targ")

f1  = @formula 0~1
f2  = @formula 0~1+Predictability+Concreteness

bfDict = Dict("fixation"=>(f1,bf1),
              "target"=>(f2,bf2))

m = Unfold.fit(UnfoldModel,
               bfDict,
               evts,
               data,asolver=(x,y) -> Unfold.solver_default(x,y;stderror=true),
                                                            eventcolumn="type")


results = coeftable(m)
#plot_erp(results)
CSV.write("/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/eeg_coeftable_$(sbj_id)_concpred.csv", results, transform=(col, val) -> something(val, missing))

eff = effects(Dict(:Predictability => ["Predictable",
                                       "Unpredictable"]),m)

CSV.write("/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/eeg_effect_$(sbj_id)_pred.csv", eff, transform=(col, val) -> something(val, missing))

eff = effects(Dict(:Concreteness => ["Concrete",
                                       "Abstract"]),m)

CSV.write("/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/eeg_effect_$(sbj_id)_conc.csv", eff, transform=(col, val) -> something(val, missing))

