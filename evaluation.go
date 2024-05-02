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

func GenerateAndCountUnique(grn *RegulatoryNetwork, n int, targets []*mat.VecDense) (numUniqueClassesSeen, numInTargets int) {
	productions := make([]*mat.VecDense, n)
	for i := 0; i < n; i++ {
		genotype := NewGenotype(grn.GenotypeSize, 0)
		productions[i] = grn.Run(genotype.Vector)
	}
	return CountUnique(productions, targets)
}

func CountUnique(produced, targets []*mat.VecDense) (numUniqueClassesSeen, numInTargets int) {
	seen := make(map[int]bool)
	for _, p := range produced {
		for ti, t := range targets {
			score := sameScore(p, t)
			if score > float64(p.Len())-3 {
				numInTargets++
				seen[ti] = true
				break
			}
		}
	}
	return len(seen), numInTargets
}

func sameScore(a, b *mat.VecDense) float64 {
	// Clamp values in a and b
	for i := 0; i < a.Len(); i++ {
		a.SetVec(i, clamp(-1, 1)(a.AtVec(i)))
		b.SetVec(i, clamp(-1, 1)(b.AtVec(i)))
	}
	result := mat.NewVecDense(a.Len(), nil)
	result.MulElemVec(a, b)
	sum := mat.Sum(result)
	return sum
}
