package main

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math/rand"
	"os"
	"path"

	"gonum.org/v1/gonum/mat"
)

func loadArbitaryDataset() ([]*mat.VecDense, int, int, error) {
	return []*mat.VecDense{mat.NewVecDense(8, []float64{1, 1, -1, -1, -1, 1, -1, 1})}, 8, 1, nil
}

func LoadDataset(dataPath string) ([]*mat.VecDense, int, int, error) {
	if dataPath == "arbitary" {
		return loadArbitaryDataset()
	}
	// Find all subfolders in the data path
	subfolders, err := os.ReadDir(dataPath)
	if err != nil {
		return nil, 0, 0, err
	}
	images := make([]*mat.VecDense, 0)
	imgSize := -1
	for _, subfolder := range subfolders {
		subfolderPath := path.Join(dataPath, subfolder.Name())
		// Find all files in the subfolder
		files, err := os.ReadDir(subfolderPath)
		if err != nil {
			return nil, 0, 0, err
		}
		for _, file := range files {
			filePath := path.Join(subfolderPath, file.Name())
			f, err := os.Open(filePath)
			if err != nil {
				return nil, 0, 0, err
			}
			defer f.Close()
			var img image.Image
			// if it is a .jpg
			if path.Ext(filePath) == ".jpg" || path.Ext(filePath) == ".jpeg" {
				img, err = jpeg.Decode(f)
				if err != nil {
					return nil, 0, 0, err
				}
			} else if path.Ext(filePath) == ".png" {
				img, err = png.Decode(f)
				if err != nil {
					return nil, 0, 0, err
				}
			} else {
				return nil, 0, 0, fmt.Errorf("unsupported file type")
			}
			if img.Bounds().Dx() != img.Bounds().Dy() {
				panic("Image is not square")
			}
			if imgSize == -1 {
				imgSize = img.Bounds().Dx()
			} else if imgSize != img.Bounds().Dx() {
				panic("Image size mismatch, all images must be the same size")
			}
			// Convert the image to a vector
			vec := Img2Vec(img)
			images = append(images, vec)
		}
	}
	rand.Shuffle(len(images), func(i, j int) {
		images[i], images[j] = images[j], images[i]
	})
	return images, imgSize, imgSize, nil
}

func Vec2Img(v *mat.VecDense, imgSizeX, imgSizeY int) image.Image {
	if v.Len() != imgSizeX*imgSizeY {
		panic("Vector length does not match image size")
	}
	img := image.NewGray(image.Rect(0, 0, imgSizeX, imgSizeY))
	for i := 0; i < imgSizeX; i++ {
		for j := 0; j < imgSizeY; j++ {
			val := v.AtVec(i*imgSizeX + j)
			val = clamp(-1, 1)(val)/2 + 0.5
			img.SetGray(i, j, color.Gray{Y: uint8(val * 255)})
		}
	}
	return img
}

func Mat2Img(m *mat.Dense, maxVal float64) image.Image {
	r, c := m.Dims()
	img := image.NewGray(image.Rect(0, 0, r, c))
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := (clamp(-maxVal, maxVal)(m.At(i, j)) + maxVal) / 2
			img.SetGray(i, j, color.Gray{Y: uint8(v / maxVal * 255)})
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
			data[i*bounds.Dy()+j] = data[i*bounds.Dy()+j]*2 - 1
		}
	}
	return mat.NewVecDense(size, data)
}
