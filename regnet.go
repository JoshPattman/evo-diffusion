package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type RegNetwork interface {
	// Run the regulatory netowkr for a number of timesteps, with the initial state given by the genotype
	Run(genotype *mat.VecDense, timesteps int) (phenotype *mat.VecDense)
	// Mutate the network at its default rate
	Mutate()
	// Copy the parameters of the other network into this one
	CopyFrom(other RegNetwork)
}

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

func (d *DenseRegNetwork) Run(genotype *mat.VecDense, timesteps int) (phenotype *mat.VecDense) {
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

func (d *DenseRegNetwork) Mutate() {
	r, c := d.Weights.Dims()
	ri := rand.Intn(r)
	ci := rand.Intn(c)
	d.Weights.Set(ri, ci, d.Weights.At(ri, ci)+d.WeightsMaxMut*(rand.Float64()*2-1))
}

func (d *DenseRegNetwork) CopyFrom(other RegNetwork) {
	otherDense := other.(*DenseRegNetwork)
	d.Weights.Copy(otherDense.Weights)
	d.UpdateRate = otherDense.UpdateRate
	d.DecayRate = otherDense.DecayRate
	d.WeightsMaxMut = otherDense.WeightsMaxMut
}
