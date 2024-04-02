package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Evaluation of 0 is best, -1 is worst
func Evaluate(g *Genotype, r RegNetwork, target *mat.VecDense, timesteps int) float64 {
	result := r.Run(g.Vector, timesteps)
	hs := imgHebbScore(result, target)
	hs = math.Abs(hs) // Perfect negative is also good
	return hs
}

func imgHebbScore(predicted, target *mat.VecDense) float64 {
	hs := mat.NewVecDense(predicted.Len(), nil)
	hs.MulElemVec(predicted, target)
	sum := mat.Sum(hs)
	return sum
}
