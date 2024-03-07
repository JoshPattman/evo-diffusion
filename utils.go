package main

import (
	"image"
	"image/png"
	"math"
	"math/rand"
	"os"

	"gonum.org/v1/gonum/mat"
)

func ApplyAllVec(v *mat.VecDense, f func(float64) float64) {
	d := v.RawVector().Data
	for i := range d {
		d[i] = f(d[i])
	}
}

func SaveImg(filename string, img image.Image) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func tanh(x float64) float64 {
	return math.Tanh(x)
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

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func relu(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func uniformVec(size int, maxVal float64) *mat.VecDense {
	data := make([]float64, size)
	for i := range data {
		data[i] = (rand.Float64()*2 - 1) * maxVal
	}
	return mat.NewVecDense(size, data)
}

func uniformMat(rows, cols int, maxVal float64) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = (rand.Float64()*2 - 1) * maxVal
	}
	return mat.NewDense(rows, cols, data)
}
