package main

import (
	"gonum.org/v1/gonum/mat"
)

type RegNetwork interface {
	// Run the regulatory netowkr for a number of timesteps, with the initial state given by the genotype
	Run(genotype *mat.VecDense, timesteps int) (finalState *mat.VecDense)
	// Same as Run, but also returns all intermediate states. Allocated more usually so will be slowe
	RunWithIntermediateStates(genotype *mat.VecDense, timesteps int) (states []*mat.VecDense)
	// Mutate the network at its default rate
	Mutate()
	// Copy the parameters of the other network into this one
	CopyFrom(other RegNetwork)
	WeightsMatrix() *mat.Dense
}
