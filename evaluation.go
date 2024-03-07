package main

import (
	"gonum.org/v1/gonum/mat"
)

// Evaluation of 0 is best, -1 is worst
func Evaluate(g *Genotype, r RegNetwork, target *mat.VecDense, timesteps int) float64 {
	result := r.Run(g.Vector, timesteps)
	mse := imgMse(result, target)
	return -mse
}

func imgMse(predicted, target *mat.VecDense) float64 {
	diff := mat.NewVecDense(predicted.Len(), nil)
	diff.SubVec(target, predicted)
	diffNorm := diff.Norm(2)
	mse := (diffNorm * diffNorm) / float64(diff.Len())
	return mse
}
