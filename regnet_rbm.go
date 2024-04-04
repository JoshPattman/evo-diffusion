package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

var _ RegNetwork = &RegNetRBM{}

type RegNetRBM struct {
	Weights            *mat.Dense
	UpdateRate         float64
	DecayRate          float64
	WeightsMaxMut      float64
	PerformWeightClamp bool
}

func NewRegNetRBM(nodes, hiddenNodes int, updateRate float64, decayRate float64, weightsMaxMut float64, performWeightClamp bool) *RegNetRBM {
	return &RegNetRBM{
		Weights:            mat.NewDense(hiddenNodes, nodes, nil),
		UpdateRate:         updateRate,
		DecayRate:          decayRate,
		WeightsMaxMut:      weightsMaxMut,
		PerformWeightClamp: performWeightClamp,
	}
}

// CopyFrom implements RegNetwork.
func (r *RegNetRBM) CopyFrom(other RegNetwork) {
	r2 := other.(*RegNetRBM)
	r.Weights.Copy(r2.Weights)
	r.UpdateRate = r2.UpdateRate
	r.DecayRate = r2.DecayRate
	r.WeightsMaxMut = r2.WeightsMaxMut
	r.PerformWeightClamp = r2.PerformWeightClamp
}

// Mutate implements RegNetwork.
func (r *RegNetRBM) Mutate() {
	rs, cs := r.Weights.Dims()
	ri, ci := rand.Intn(rs), rand.Intn(cs)
	newVal := r.Weights.At(ri, ci) + (rand.Float64()*2-1)*r.WeightsMaxMut
	if r.PerformWeightClamp {
		newVal = clamp(-1, 1)(newVal)
	}
	r.Weights.Set(ri, ci, newVal)
}

// Run implements RegNetwork.
func (r *RegNetRBM) Run(genotype *mat.VecDense, timesteps int) (finalState *mat.VecDense) {
	state := mat.VecDenseCopyOf(genotype)
	stateUpdate := mat.NewVecDense(state.Len(), nil)
	hiddenState := mat.NewVecDense(r.Weights.RawMatrix().Rows, nil)
	for i := 0; i < timesteps; i++ {
		hiddenState.MulVec(r.Weights, state)
		ApplyAllVec(hiddenState, tanh)
		// Multiply weights by state and apply tanh
		stateUpdate.MulVec(r.Weights.T(), hiddenState)
		ApplyAllVec(stateUpdate, tanh)
		// Decay the state
		state.ScaleVec(1-r.DecayRate, state)
		// Add the update to the state
		state.AddScaledVec(state, r.UpdateRate, stateUpdate)
	}
	return state
}

// RunWithIntermediateStates implements RegNetwork.
func (r *RegNetRBM) RunWithIntermediateStates(genotype *mat.VecDense, timesteps int) (states []*mat.VecDense) {
	states = make([]*mat.VecDense, timesteps+1)
	state := mat.VecDenseCopyOf(genotype)
	states[0] = mat.VecDenseCopyOf(state)
	stateUpdate := mat.NewVecDense(state.Len(), nil)
	hiddenState := mat.NewVecDense(r.Weights.RawMatrix().Rows, nil)
	for i := 0; i < timesteps; i++ {
		hiddenState.MulVec(r.Weights, state)
		ApplyAllVec(hiddenState, tanh)
		// Multiply weights by state and apply tanh
		stateUpdate.MulVec(r.Weights.T(), hiddenState)
		ApplyAllVec(stateUpdate, tanh)
		// Decay the state
		state.ScaleVec(1-r.DecayRate, state)
		// Add the update to the state
		state.AddScaledVec(state, r.UpdateRate, stateUpdate)
		states[i+1] = mat.VecDenseCopyOf(state)
	}
	return states
}

// WeightsMatrix implements RegNetwork.
func (r *RegNetRBM) WeightsMatrix() *mat.Dense {
	return r.Weights
}
