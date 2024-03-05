package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

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
