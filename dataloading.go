package main

import (
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"math/rand"
	"os"
	"path"

	"gonum.org/v1/gonum/mat"
)

func LoadDataset(dataPath string) ([]*mat.VecDense, error) {
	// Find all subfolders in the data path
	subfolders, err := os.ReadDir(dataPath)
	if err != nil {
		return nil, err
	}
	images := make([]*mat.VecDense, 0)
	for _, subfolder := range subfolders {
		subfolderPath := path.Join(dataPath, subfolder.Name())
		// Find all files in the subfolder
		files, err := os.ReadDir(subfolderPath)
		if err != nil {
			return nil, err
		}
		for _, file := range files {
			filePath := path.Join(subfolderPath, file.Name())
			f, err := os.Open(filePath)
			if err != nil {
				return nil, err
			}
			defer f.Close()
			img, err := jpeg.Decode(f)
			if err != nil {
				return nil, err
			}
			// Convert the image to a vector
			vec := Img2Vec(img)
			images = append(images, vec)
		}
	}
	rand.Shuffle(len(images), func(i, j int) {
		images[i], images[j] = images[j], images[i]
	})
	return images, nil
}

func Vec2Img(v *mat.VecDense) image.Image {
	fsqrt := math.Sqrt(float64(v.Len()))
	sqrt := int(fsqrt)
	if fsqrt != float64(sqrt) {
		panic("Vector length is not a square number")
	}
	img := image.NewGray(image.Rect(0, 0, sqrt, sqrt))
	for i := 0; i < sqrt; i++ {
		for j := 0; j < sqrt; j++ {
			val := v.AtVec(i*sqrt + j)
			val = clamp(0, 1)(val)
			img.SetGray(i, j, color.Gray{Y: uint8(val * 255)})
		}
	}
	return img
}

func Mat2Img(m *mat.Dense) image.Image {
	r, c := m.Dims()
	img := image.NewGray(image.Rect(0, 0, r, c))
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := sigmoid(m.At(i, j)) // Apply sigmoid here to map matrix values into nice range
			v = clamp(0, 1)(v)
			img.SetGray(i, j, color.Gray{Y: uint8(v * 255)})
		}
	}
	return img
}

func Img2Vec(img image.Image) *mat.VecDense {
	bounds := img.Bounds()
	size := bounds.Dx() * bounds.Dy()
	data := make([]float64, size)
	for i := 0; i < bounds.Dx(); i++ {
		for j := 0; j < bounds.Dy(); j++ {
			data[i*bounds.Dy()+j] = float64(color.GrayModel.Convert(img.At(i, j)).(color.Gray).Y) / 255
			//data[i*bounds.Dy()+j] = data[i*bounds.Dy()+j]*2 - 1
		}
	}
	return mat.NewVecDense(size, data)
}
