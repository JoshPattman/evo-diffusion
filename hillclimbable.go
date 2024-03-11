package main

import (
	"gonum.org/v1/gonum/mat"
)

type HillClimbable interface {
	// Create a deep exact copy of this HillClimbable
	Clone() HillClimbable
	// Modify this hillclimbable to be a mutated version of other. Must be same shaped.
	MutateFrom(other HillClimbable)
	// Modify this hillclimbable to be an average of others. Must be same shaped.
	AverageFrom(others []HillClimbable)
}

type HCRegNet interface {
	HillClimbable
	// Run the regulatory netowkr for a number of timesteps, with the initial state given by the genotype
	Run(genotype *mat.VecDense, timesteps int) (finalState *mat.VecDense)
	// Same as Run, but also returns all intermediate states. Allocated more usually so will be slowe
	RunWithIntermediateStates(genotype *mat.VecDense, timesteps int) (states []*mat.VecDense)
}
