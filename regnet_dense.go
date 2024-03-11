package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

var _ HCRegNet = &DenseRegNetwork{}

func NewDenseRegNetwork(nodes int, updateRate float64, decayRate float64, weightsMaxMult float64, numMutations int, chanceOfMutation float64) *DenseRegNetwork {
	return &DenseRegNetwork{
		Weights:          mat.NewDense(nodes, nodes, nil),
		UpdateRate:       updateRate,
		DecayRate:        decayRate,
		WeightsMaxMut:    weightsMaxMult,
		NumMutations:     numMutations,
		ChanceOfMutation: chanceOfMutation,
	}
}

type DenseRegNetwork struct {
	// The weights between the nodes in the network
	Weights *mat.Dense
	// The rate at which the state is updated
	UpdateRate float64
	// The rate at which the state decays
	DecayRate float64
	// The maximum amount by which the weights can be mutated
	WeightsMaxMut    float64
	NumMutations     int
	ChanceOfMutation float64
}

// Clone implements RegNetwork.
func (n *DenseRegNetwork) Clone() HillClimbable {
	return &DenseRegNetwork{
		Weights:          mat.DenseCopyOf(n.Weights),
		UpdateRate:       n.UpdateRate,
		DecayRate:        n.DecayRate,
		WeightsMaxMut:    n.WeightsMaxMut,
		NumMutations:     n.NumMutations,
		ChanceOfMutation: n.ChanceOfMutation,
	}
}

// MutateFrom implements HillClimbable.
func (n *DenseRegNetwork) MutateFrom(other HillClimbable) {
	// Copy into this
	n.Weights.Copy(other.(*Genotype).Vector)

	// Mutate n times
	if rand.Float64() < n.ChanceOfMutation {
		r, c := n.Weights.Dims()
		for i := 0; i < n.NumMutations; i++ {
			ri, ci := rand.Intn(r), rand.Intn(c)
			addition := n.WeightsMaxMut * (rand.Float64()*2 - 1)
			n.Weights.Set(ri, ci, n.Weights.At(ri, ci)+addition)
		}
	}
}

// AverageFrom implements HillClimbable.
func (n *DenseRegNetwork) AverageFrom(others []HillClimbable) {
	n.Weights.Zero()
	for _, o := range others {
		n.Weights.Add(n.Weights, o.(*DenseRegNetwork).Weights)
	}
	n.Weights.Scale(1.0/float64(len(others)), n.Weights)
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
