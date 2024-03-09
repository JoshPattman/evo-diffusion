package main

import "math/rand"

type GenoRegPair struct {
	Genotype   *Genotype
	RegNetwork RegNetwork
}

type GRReproduction struct {
	RegNetMutateChance float64
}

func (gr *GRReproduction) Reproduce(a, b *GenoRegPair) *GenoRegPair {
	grp := &GenoRegPair{
		a.Genotype.CrossoverWith(b.Genotype),
		a.RegNetwork.CrossoverWith(b.RegNetwork),
	}
	grp.Genotype.Mutate()
	if rand.Float64() < gr.RegNetMutateChance {
		grp.RegNetwork.Mutate()
	}
	return grp
}
