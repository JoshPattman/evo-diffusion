package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Evaluation of 0 is best, -1 is worst
func Evaluate(g *Genotype, target *mat.VecDense, l2 float64) float64 {
	result := g.Generate()
	diff := mat.NewVecDense(result.Len(), nil)
	diff.SubVec(target, result)
	diffNorm := diff.Norm(2)
	mse := (diffNorm * diffNorm) / float64(diff.Len())
	wmNorm := g.Matrix.Norm(2)
	r, c := g.Matrix.Dims()
	ml2 := (wmNorm * wmNorm) / float64(r*c)
	return -mse - l2*ml2
}

func NewRandTarVec(size int) *mat.VecDense {
	// return a vec where elements with idx < 0.5*size are 1, and 0 otherwise
	modifier := rand.Float64() > 0.5
	data := make([]float64, size)
	for i := range data {
		if modifier {
			if i < size/2 {
				data[i] = 1
			}
		} else {
			if i >= size/2 {
				data[i] = 1
			}
		}
	}
	return mat.NewVecDense(size, data)
}
