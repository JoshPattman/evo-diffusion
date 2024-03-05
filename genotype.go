package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Genotype struct {
	matrix     *mat.Dense
	vector     *mat.VecDense
	updateRate float64 // tau1
	decayRate  float64 // tau2
	iterations int
}

func NewGenotype(size int) *Genotype {
	return &Genotype{
		matrix:     mat.NewDense(size, size, nil),
		vector:     mat.NewVecDense(size, nil),
		updateRate: 1,
		decayRate:  0.2,
		iterations: 10,
	}
}

func (g *Genotype) Generate() *mat.VecDense {
	result := mat.VecDenseCopyOf(g.vector)
	tempResult := mat.NewVecDense(result.Len(), nil)
	for i := 0; i < g.iterations; i++ {
		tempResult.MulVec(g.matrix, result)
		ApplyAllVec(tempResult, tanh)
		// At this point, tempResult = t_1 * a(BxP(t))
		tempResult.ScaleVec(g.updateRate, tempResult)
		// At this point, result = P(t) - t_2 * P(t) = (1-t_2) * P(t)
		result.ScaleVec(1-g.decayRate, result)
		// Now, result is the final sum
		result.AddVec(result, tempResult)
		// Clamp to -1-1
		ApplyAllVec(result, clamp(0, 1))
	}
	return result
}

func (g *Genotype) Mutate(matrixAmount, matrixProb, vectorAmount float64) {
	if rand.Float64() < matrixProb {
		r, c := g.matrix.Dims()
		ri, ci := rand.Intn(r), rand.Intn(c)
		g.matrix.Set(ri, ci, g.matrix.At(ri, ci)+matrixAmount*(rand.Float64()*2-1))
	}

	d := g.vector.Len()
	di := rand.Intn(d)
	g.vector.SetVec(di, clamp(0, 1)(g.vector.AtVec(di)+vectorAmount*(rand.Float64()*2-1)))
}

func (g *Genotype) CopySrcVec(src *mat.VecDense) {
	g.vector.CopyVec(src)
}

func (g *Genotype) CopyGenotypeFrom(other *Genotype) {
	g.matrix.Copy(other.matrix)
	g.vector.CopyVec(other.vector)
}

func NewSrcVec(size int) *mat.VecDense {
	vecData := make([]float64, size)
	for i := 0; i < size; i++ {
		vecData[i] = rand.Float64()
	}
	return mat.NewVecDense(size, vecData)
}
