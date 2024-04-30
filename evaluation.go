package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Evaluation of 0 is best, -1 is worst
func Evaluate(g *Genotype, r *RegulatoryNetwork, target *mat.VecDense, l2Fac float64) float64 {
	result := r.Run(g.Vector)
	hs := imgHebbScore(result, target)
	hs = math.Abs(hs) // Perfect negative is also good
	l2Val := mat.Norm(r.WeightsMatrix(), 2)
	return hs - l2Fac*l2Val
}

func imgHebbScore(predicted, target *mat.VecDense) float64 {
	hs := mat.NewVecDense(predicted.Len(), nil)
	hs.MulElemVec(predicted, target)
	/*for i := 0; i < hs.Len(); i++ {
		hs.SetVec(i, clamp(-1, 1)(hs.AtVec(i)))
	}*/
	sum := mat.Sum(hs)
	return sum
}
