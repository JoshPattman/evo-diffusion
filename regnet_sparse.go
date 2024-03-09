package main

import (
	"math/rand"
	"slices"

	"gonum.org/v1/gonum/mat"
)

var _ RegNetwork = &SparseRegNetwork{}

func NewSparseRegNetwork(numNodes, numWeights int, updateRate, decayRate, mutWeightAmount, moveConProb float64) *SparseRegNetwork {
	ws := make([]*SparseWeight, numWeights)
	for i := range ws {
		ws[i] = &SparseWeight{
			From:   rand.Intn(numNodes),
			To:     rand.Intn(numNodes),
			Weight: 0,
		}
	}
	return &SparseRegNetwork{
		Weights:            ws,
		UpdateRate:         updateRate,
		DecayRate:          decayRate,
		MutWeightAmount:    mutWeightAmount,
		NumNodes:           numNodes,
		MoveConnectionProb: moveConProb,
	}
}

type SparseWeight struct {
	From   int
	To     int
	Weight float64
}

type SparseRegNetwork struct {
	Weights []*SparseWeight
	// The rate at which the state is updated
	UpdateRate float64
	// The rate at which the state decays
	DecayRate          float64
	MutWeightAmount    float64
	MoveConnectionProb float64
	NumNodes           int
}

// WeightsMatrix implements RegNetwork.
func (n *SparseRegNetwork) WeightsMatrix() *mat.Dense {
	weights := mat.NewDense(n.NumNodes, n.NumNodes, nil)
	for _, w := range n.Weights {
		weights.Set(w.From, w.To, w.Weight)
	}
	return weights
}

// CopyFrom implements RegNetwork.
func (n *SparseRegNetwork) CopyFrom(other RegNetwork) {
	n.Weights = slices.Clone(other.(*SparseRegNetwork).Weights)
}

// Mutate implements RegNetwork.
func (n *SparseRegNetwork) Mutate() {
	n.Weights[rand.Intn(len(n.Weights))].Weight += n.MutWeightAmount * (rand.Float64()*2 - 1)
	if rand.Float64() < n.MoveConnectionProb {
		w := n.Weights[rand.Intn(len(n.Weights))]
		if rand.Float64() < 0.5 {
			w.From = rand.Intn(n.NumNodes)
		} else {
			w.To = rand.Intn(n.NumNodes)
		}
		w.Weight = (2*rand.Float64() - 1) * n.MutWeightAmount
	}
}

// Run implements RegNetwork.
func (n *SparseRegNetwork) Run(genotype *mat.VecDense, timesteps int) (finalState *mat.VecDense) {
	state := make([]float64, genotype.Len())
	updateState := make([]float64, genotype.Len())
	for i := range state {
		state[i] = genotype.AtVec(i)
	}
	for t := 0; t < timesteps; t++ {
		// Clear update state
		for i := range updateState {
			updateState[i] = 0
		}
		for _, w := range n.Weights {
			updateState[w.To] += w.Weight * state[w.From]
		}
		for i := range state {
			state[i] = (1-n.DecayRate)*state[i] + n.UpdateRate*tanh(updateState[i])
		}
	}
	return mat.NewVecDense(len(state), state)
}

// RunWithIntermediateStates implements RegNetwork.
func (n *SparseRegNetwork) RunWithIntermediateStates(genotype *mat.VecDense, timesteps int) []*mat.VecDense {
	states := make([]*mat.VecDense, timesteps+1)
	state := make([]float64, genotype.Len())
	updateState := make([]float64, genotype.Len())
	for i := range state {
		state[i] = genotype.AtVec(i)
	}
	scopy := make([]float64, genotype.Len())
	copy(scopy, state)
	states[0] = mat.NewVecDense(len(state), scopy)
	for t := 0; t < timesteps; t++ {
		// Clear update state
		for i := range updateState {
			updateState[i] = 0
		}
		for _, w := range n.Weights {
			updateState[w.To] += w.Weight * state[w.From]
		}
		for i := range state {
			state[i] = (1-n.DecayRate)*state[i] + n.UpdateRate*tanh(updateState[i])
		}
		scopy := make([]float64, genotype.Len())
		copy(scopy, state)
		states[t+1] = mat.NewVecDense(len(state), scopy)
	}
	return states
}
