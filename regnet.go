package main

import (
	"gonum.org/v1/gonum/mat"
)

type RegNetwork interface {
	// Run the regulatory netowkr for a number of timesteps, with the initial state given by the genotype
	Run(genotype *mat.VecDense, timesteps int) (finalState *mat.VecDense)
	// Same as Run, but also returns all intermediate states. Allocated more usually so will be slowe
	RunWithIntermediateStates(genotype *mat.VecDense, timesteps int) (states []*mat.VecDense)
	// Clone (asexual reproduction)
	Clone() RegNetwork
	// Crossover (sexual reproduction)
	CrossoverWith(other RegNetwork) RegNetwork
	// Mutate the network at its default rate
	Mutate()
	// Generate a metrix representing the weights - only used for drawing
	WeightsMatrix() *mat.Dense
}
