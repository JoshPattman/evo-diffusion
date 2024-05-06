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
			numIncorrect := p.Len() - numCorrect(p, t)
			if numIncorrect == 0 || numIncorrect == p.Len() {
				numInTargets++
				seen[ti] = true
				break
			}
		}
	}
	return len(seen), numInTargets
}

func numCorrect(a, b *mat.VecDense) int {
	total := 0
	for i := 0; i < a.Len(); i++ {
		if math.Signbit(a.AtVec(i)) == math.Signbit(b.AtVec(i)) {
			total++
		}
	}
	return total
}
