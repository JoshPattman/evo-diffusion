package main

import (
	"math"
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

func NewGenotype(size int, maxWeightInit float64) *Genotype {
	matrixData := make([]float64, size*size)
	for i := 0; i < size*size; i++ {
		matrixData[i] = maxWeightInit * (rand.Float64()*2 - 1)
	}
	return &Genotype{
		matrix:     mat.NewDense(size, size, matrixData),
		vector:     mat.NewVecDense(size, nil),
		updateRate: 1,
		decayRate:  0.2,
		iterations: 5,
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
		// NOT THE SAME AS IN THE PAPER:
		ApplyAllVec(result, clamp(0, 1))
	}
	return result
}

func tanh(x float64) float64 {
	return math.Tanh(x)
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
func clamp(min, max float64) func(float64) float64 {
	return func(x float64) float64 {
		if x < min {
			return min
		} else if x > max {
			return max
		}
		return x
	}
}

func (g *Genotype) Mutate(matrixRate, matrixAmount, vectorRate, vectorAmount, zeroMatProb float64) {
	r, c := g.matrix.Dims()
	matrixData := make([]float64, r*c)
	for i := range matrixData {
		if rand.Float64() < matrixRate {
			matrixData[i] = matrixAmount * (rand.Float64()*2 - 1)
		}
	}

	d := g.vector.Len()
	vectorData := make([]float64, d)
	for i := range vectorData {
		if rand.Float64() < vectorRate {
			vectorData[i] = vectorAmount * (rand.Float64()*2 - 1)
		}
	}

	g.matrix.Add(g.matrix, mat.NewDense(r, c, matrixData))

	g.vector.AddVec(g.vector, mat.NewVecDense(d, vectorData))
	ApplyAllVec(g.vector, clamp(0, 1))
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

func ApplyAllVec(v *mat.VecDense, f func(float64) float64) {
	for i := range v.RawVector().Data {
		v.RawVector().Data[i] = f(v.RawVector().Data[i])

	}
}

func ApplyAllMat(m *mat.Dense, f func(float64) float64) {
	for i := range m.RawMatrix().Data {
		m.RawMatrix().Data[i] = f(m.RawMatrix().Data[i])
	}
}
