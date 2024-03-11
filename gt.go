package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func NewGenotype(length int, valsMaxMut float64, numMutations int) *Genotype {
	backing := make([]float64, length)
	for i := range backing {
		backing[i] = rand.Float64()
	}
	return &Genotype{
		Vector:       mat.NewVecDense(length, backing),
		ValsMaxMut:   valsMaxMut,
		NumMutations: numMutations,
	}
}

type Genotype struct {
	Vector       *mat.VecDense
	ValsMaxMut   float64
	NumMutations int
}

func (gt *Genotype) Clone() HillClimbable {
	return &Genotype{
		Vector:       mat.VecDenseCopyOf(gt.Vector),
		ValsMaxMut:   gt.ValsMaxMut,
		NumMutations: gt.NumMutations,
	}
}

func (gt *Genotype) MutateFrom(other HillClimbable) {
	// Copy into this
	gt.Vector.CopyVec(other.(*Genotype).Vector)

	// Mutate n times
	d := gt.Vector.Len()
	for i := 0; i < gt.NumMutations; i++ {
		di := rand.Intn(d)
		addition := gt.ValsMaxMut * (rand.Float64()*2 - 1)
		gt.Vector.SetVec(di, gt.Vector.AtVec(di)+addition)
	}
}

func (gt *Genotype) AverageFrom(others []HillClimbable) {
	gt.Vector.Zero()
	for _, o := range others {
		gt.Vector.AddVec(gt.Vector, o.(*Genotype).Vector)
	}
	gt.Vector.ScaleVec(1.0/float64(len(others)), gt.Vector)
}
