package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

var _ RegNetwork = &DenseRegNetwork{}

type DenseRegNetwork struct {
	// The weights between the nodes in the network
	Weights *mat.Dense
	// The rate at which the state is updated
	UpdateRate float64
	// The rate at which the state decays
	DecayRate float64
	// The maximum amount by which the weights can be mutated
	WeightsMaxMut float64
}

func NewDenseRegNetwork(nodes int, updateRate float64, decayRate float64, weightsMaxMult float64) *DenseRegNetwork {
	return &DenseRegNetwork{
		Weights:       mat.NewDense(nodes, nodes, nil),
		UpdateRate:    updateRate,
		DecayRate:     decayRate,
		WeightsMaxMut: weightsMaxMult,
	}
}

func (d *DenseRegNetwork) Run(genotype *mat.VecDense, timesteps int) *mat.VecDense {
	state := mat.VecDenseCopyOf(genotype)
	stateUpdate := mat.NewVecDense(state.Len(), nil)
	for i := 0; i < timesteps; i++ {
		// Multiply weights by state and apply tanh
		stateUpdate.MulVec(d.Weights, state)
		ApplyAllVec(stateUpdate, tanh)
		// Decay the state
		state.ScaleVec(1-d.DecayRate, state)
		// Add the update to the state
		state.AddScaledVec(state, d.UpdateRate, stateUpdate)
		// Ensure the state is still in range -1 to 1
		ApplyAllVec(state, clamp(0, 1))
	}
	return state
}

func (d *DenseRegNetwork) RunWithIntermediateStates(genotype *mat.VecDense, timesteps int) []*mat.VecDense {
	states := make([]*mat.VecDense, timesteps+1)
	state := mat.VecDenseCopyOf(genotype)
	states[0] = mat.VecDenseCopyOf(state)
	stateUpdate := mat.NewVecDense(state.Len(), nil)
	for i := 0; i < timesteps; i++ {
		// Multiply weights by state and apply tanh
		stateUpdate.MulVec(d.Weights, state)
		ApplyAllVec(stateUpdate, tanh)
		// Decay the state
		state.ScaleVec(1-d.DecayRate, state)
		// Add the update to the state
		state.AddScaledVec(state, d.UpdateRate, stateUpdate)
		// Ensure the state is still in range -1 to 1
		ApplyAllVec(state, clamp(0, 1))
		states[i+1] = mat.VecDenseCopyOf(state)
	}
	return states
}

func (n *DenseRegNetwork) Clone() RegNetwork {
	return &DenseRegNetwork{
		Weights:       mat.DenseCopyOf(n.Weights),
		UpdateRate:    n.UpdateRate,
		DecayRate:     n.DecayRate,
		WeightsMaxMut: n.WeightsMaxMut,
	}
}

func (n *DenseRegNetwork) CrossoverWith(other RegNetwork) RegNetwork {
	r, c := n.Weights.Dims()
	weights := mat.NewDense(r, c, nil)
	for ri := 0; ri < r; ri++ {
		for ci := 0; ci < c; ci++ {
			if rand.Float64() < 0.5 {
				weights.Set(ri, ci, n.Weights.At(ri, ci))
			} else {
				weights.Set(ri, ci, other.(*DenseRegNetwork).Weights.At(ri, ci))
			}
		}
	}
	return &DenseRegNetwork{
		Weights:       weights,
		UpdateRate:    n.UpdateRate,
		DecayRate:     n.DecayRate,
		WeightsMaxMut: n.WeightsMaxMut,
	}
}

func (d *DenseRegNetwork) Mutate() {
	r, c := d.Weights.Dims()
	ri := rand.Intn(r)
	ci := rand.Intn(c)
	addition := d.WeightsMaxMut * (rand.Float64()*2 - 1)
	d.Weights.Set(ri, ci, d.Weights.At(ri, ci)+addition)
}

func (d *DenseRegNetwork) WeightsMatrix() *mat.Dense {
	return d.Weights
}
