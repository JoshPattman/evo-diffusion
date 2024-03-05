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
	diff.MulElemVec(diff, diff)
	mse := mat.Sum(diff) / float64(diff.Len())
	wm := mat.NewDense(g.matrix.RawMatrix().Rows, g.matrix.RawMatrix().Cols, nil)
	wm.MulElem(g.matrix, g.matrix)
	ml2 := mat.Sum(wm) / float64(wm.RawMatrix().Rows*wm.RawMatrix().Cols)
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
