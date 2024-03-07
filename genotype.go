package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func NewGenotype(length int, maxMut float64) *Genotype {
	data := make([]float64, length)
	for i := range data {
		data[i] = rand.Float64()
	}
	return &Genotype{
		ValsMaxMut: maxMut,
		Vector:     mat.NewVecDense(length, data),
	}
}

// Genotype wraps a vector that represents the initial state of a regulatory network
// It can be mutated and copied
type Genotype struct {
	// The state vector
	Vector *mat.VecDense
	// The maximum amount by which the values in the vector can be mutated
	ValsMaxMut float64
}

func (g *Genotype) Mutate() {
	d := g.Vector.Len()
	di := rand.Intn(d)
	g.Vector.SetVec(di, g.Vector.AtVec(di)+g.ValsMaxMut*(rand.Float64()*2-1))
}

func (g *Genotype) CopyFrom(other *Genotype) {
	g.Vector.CopyVec(other.Vector)
	g.ValsMaxMut = other.ValsMaxMut
}
