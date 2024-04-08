package main

import (
	"gonum.org/v1/gonum/mat"
)

type TransferFunc interface {
	Transfer(state, into *mat.VecDense)
	Mutate()
	CopyFrom(other TransferFunc)
	CombinedWeightsMatrix() *mat.Dense
}

type RegulatoryNetwork struct {
	TransferFunc TransferFunc
	UpdateRate   float64
	DecayRate    float64
	Timesteps    int
	GenotypeSize int
}

func NewRegulatoryNetwork(transferFunc TransferFunc, updateRate, decayRate float64, timesteps, genotypeSize int) *RegulatoryNetwork {
	return &RegulatoryNetwork{
		TransferFunc: transferFunc,
		UpdateRate:   updateRate,
		DecayRate:    decayRate,
		Timesteps:    timesteps,
		GenotypeSize: genotypeSize,
	}
}

// CopyFrom implements RegNetwork.
func (t *RegulatoryNetwork) CopyFrom(other *RegulatoryNetwork) {
	t.TransferFunc.CopyFrom(other.TransferFunc)
	t.UpdateRate = other.UpdateRate
	t.DecayRate = other.DecayRate
	t.Timesteps = other.Timesteps
	t.GenotypeSize = other.GenotypeSize
}

// Mutate implements RegNetwork.
func (t *RegulatoryNetwork) Mutate() {
	t.TransferFunc.Mutate()
}

// Run implements RegNetwork.
func (t *RegulatoryNetwork) Run(genotype *mat.VecDense) (finalState *mat.VecDense) {
	if genotype.Len() != t.GenotypeSize {
		panic("genotype size does not match")
	}
	state := mat.VecDenseCopyOf(genotype)
	stateUpdate := mat.NewVecDense(t.GenotypeSize, nil)
	for i := 0; i < t.Timesteps; i++ {
		// Multiply weights by state and apply tanh
		t.TransferFunc.Transfer(state, stateUpdate)
		ApplyAllVec(stateUpdate, tanh)
		// Decay the state
		state.ScaleVec(1-t.DecayRate, state)
		// Add the update to the state
		state.AddScaledVec(state, t.UpdateRate, stateUpdate)
	}
	return state
}

// RunWithIntermediateStates implements RegNetwork.
func (t *RegulatoryNetwork) RunWithIntermediateStates(genotype *mat.VecDense) (states []*mat.VecDense) {
	if genotype.Len() != t.GenotypeSize {
		panic("genotype size does not match")
	}
	states = make([]*mat.VecDense, t.Timesteps+1)
	state := mat.VecDenseCopyOf(genotype)
	states[0] = mat.VecDenseCopyOf(state)
	stateUpdate := mat.NewVecDense(t.GenotypeSize, nil)
	for i := 0; i < t.Timesteps; i++ {
		states[i] = mat.VecDenseCopyOf(state)
		// Multiply weights by state and apply tanh
		t.TransferFunc.Transfer(state, stateUpdate)
		ApplyAllVec(stateUpdate, tanh)
		// Decay the state
		state.ScaleVec(1-t.DecayRate, state)
		// Add the update to the state
		state.AddScaledVec(state, t.UpdateRate, stateUpdate)
	}
	states[t.Timesteps] = mat.VecDenseCopyOf(state)
	return states
}

// WeightsMatrix implements RegNetwork.
func (t *RegulatoryNetwork) WeightsMatrix() *mat.Dense {
	return t.TransferFunc.CombinedWeightsMatrix()
}
