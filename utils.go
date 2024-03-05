package main

import (
	"image"
	"image/png"
	"math"
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

/*func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}*/

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
