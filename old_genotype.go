package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type OldGenotype struct {
	Matrix     *mat.Dense
	Vector     *mat.VecDense
	UpdateRate float64 // tau1
	DecayRate  float64 // tau2
	Iterations int
}

func NewGenotype(size int) *OldGenotype {
	return &OldGenotype{
		Matrix:     mat.NewDense(size, size, nil),
		Vector:     mat.NewVecDense(size, nil),
		UpdateRate: 1,
		DecayRate:  0.2,
		Iterations: 10,
	}
}

func (g *OldGenotype) Generate() *mat.VecDense {
	result := mat.VecDenseCopyOf(g.Vector)
	tempResult := mat.NewVecDense(result.Len(), nil)
	for i := 0; i < g.Iterations; i++ {
		tempResult.MulVec(g.Matrix, result)
		ApplyAllVec(tempResult, tanh)
		// At this point, tempResult = t_1 * a(BxP(t))
		tempResult.ScaleVec(g.UpdateRate, tempResult)
		// At this point, result = P(t) - t_2 * P(t) = (1-t_2) * P(t)
		result.ScaleVec(1-g.DecayRate, result)
		// Now, result is the final sum
		result.AddVec(result, tempResult)
		// Clamp to -1-1
		ApplyAllVec(result, clamp(0, 1))
	}
	return result
}

func (g *OldGenotype) GenerateWithIntermediate() []*mat.VecDense {
	results := make([]*mat.VecDense, g.Iterations+1)
	result := mat.VecDenseCopyOf(g.Vector)
	results[0] = mat.VecDenseCopyOf(result)
	tempResult := mat.NewVecDense(result.Len(), nil)
	for i := 0; i < g.Iterations; i++ {
		tempResult.MulVec(g.Matrix, result)
		ApplyAllVec(tempResult, tanh)
		// At this point, tempResult = t_1 * a(BxP(t))
		tempResult.ScaleVec(g.UpdateRate, tempResult)
		// At this point, result = P(t) - t_2 * P(t) = (1-t_2) * P(t)
		result.ScaleVec(1-g.DecayRate, result)
		// Now, result is the final sum
		result.AddVec(result, tempResult)
		// Clamp to -1-1
		ApplyAllVec(result, clamp(0, 1))
		results[i+1] = mat.VecDenseCopyOf(result)
	}
	return results
}

func (g *OldGenotype) Mutate(matrixAmount, matrixProb, vectorAmount float64) {
	if rand.Float64() < matrixProb {
		r, c := g.Matrix.Dims()
		ri, ci := rand.Intn(r), rand.Intn(c)
		g.Matrix.Set(ri, ci, g.Matrix.At(ri, ci)+matrixAmount*(rand.Float64()*2-1))
	}

	d := g.Vector.Len()
	di := rand.Intn(d)
	g.Vector.SetVec(di, clamp(0, 1)(g.Vector.AtVec(di)+vectorAmount*(rand.Float64()*2-1)))
}

func (g *OldGenotype) CopySrcVec(src *mat.VecDense) {
	g.Vector.CopyVec(src)
}

func (g *OldGenotype) CopyGenotypeFrom(other *OldGenotype) {
	g.Matrix.Copy(other.Matrix)
	g.Vector.CopyVec(other.Vector)
}

func NewSrcVec(size int) *mat.VecDense {
	vecData := make([]float64, size)
	for i := 0; i < size; i++ {
		vecData[i] = rand.Float64()
	}
	return mat.NewVecDense(size, vecData)
}
