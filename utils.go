package main

import (
	"image"
	"image/color"
	"image/draw"
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

func GenerateIntermediateDiagram(r RegNetwork, rows, trainingTimesteps, timesteps, imgSize int) image.Image {
	imgVolume := imgSize * imgSize
	resultss := make([][]*mat.VecDense, rows)
	for i := range resultss {
		resultss[i] = r.RunWithIntermediateStates(NewGenotype(imgVolume, 0).Vector, timesteps)
	}

	img := image.NewRGBA(image.Rect(0, 0, 1+(imgSize+1)*(timesteps+1), 1+(imgSize+1)*rows))
	draw.Draw(img, image.Rect(0, 0, (1+imgSize)*trainingTimesteps, img.Bounds().Max.Y), &image.Uniform{color.RGBA{0, 255, 0, 255}}, image.Point{}, draw.Src)
	draw.Draw(img, image.Rect((1+imgSize)*trainingTimesteps, 0, img.Bounds().Max.X, img.Bounds().Max.Y), &image.Uniform{color.RGBA{0, 0, 255, 255}}, image.Point{}, draw.Src)
	for irow, results := range resultss {
		for icol, res := range results {
			draw.Draw(img, image.Rect(1+icol*(imgSize+1), 1+irow*(imgSize+1), 1+(icol+1)*(imgSize+1), 1+(irow+1)*(imgSize+1)), Vec2Img(res, imgSize, imgSize), image.Point{}, draw.Src)
		}
	}
	return img
}

func GenerateBestiaryDiagram(r RegNetwork, rows, cols, timesteps, imgSize int) image.Image {
	imgVolume := imgSize * imgSize

	img := image.NewRGBA(image.Rect(0, 0, 1+(imgSize+1)*cols, 1+(imgSize+1)*rows))
	draw.Draw(img, img.Bounds(), &image.Uniform{color.RGBA{0, 0, 255, 255}}, image.Point{}, draw.Src)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			res := r.Run(NewGenotype(imgVolume, 0).Vector, timesteps)
			draw.Draw(img, image.Rect(1+j*(imgSize+1), 1+i*(imgSize+1), 1+(j+1)*(imgSize+1), 1+(i+1)*(imgSize+1)), Vec2Img(res, imgSize, imgSize), image.Point{}, draw.Src)
		}
	}
	return img
}

func GenerateFig12EDiagram(r RegNetwork, rows, timesteps, vecLength int) image.Image {
	results := make([]*mat.VecDense, rows)
	for i := range results {
		results[i] = r.Run(NewGenotype(vecLength, 0).Vector, timesteps)
	}

	img := image.NewRGBA(image.Rect(0, 0, vecLength, rows))
	for irow, res := range results {
		draw.Draw(img, image.Rect(0, irow, vecLength, irow+1), Vec2Img(res, vecLength, 1), image.Point{}, draw.Src)
	}
	return img
}
