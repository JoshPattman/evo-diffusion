package main

import (
	"fmt"
	"image"
	"image/png"
	"math/rand"
	"os"
	"time"
)

func main() {
	iterations := 2000000000
	subiterations := 100
	l2Norm := 200.0
	imgSize := 10
	logEvery := 200
	datasetPath := "./dataset-simple"

	imgVolume := imgSize * imgSize

	images, err := LoadDataset(datasetPath)
	if err != nil {
		panic(err)
	}
	fmt.Println("Loaded", len(images), "images")

	genBest := NewGenotype(imgVolume)
	genTest := NewGenotype(imgVolume)
	startTime := time.Now()
	for it := 0; it < iterations; it++ {
		if time.Since(startTime) > 8*time.Hour {
			break
		}

		//tar := images[rand.Intn(len(images))]
		tar := images[rand.Intn(1)]
		src := NewSrcVec(imgVolume)

		genTest.CopySrcVec(src)
		genBest.CopySrcVec(src)
		genBestEval := Evaluate(genBest, tar, l2Norm)
		for sit := 0; sit < subiterations; sit++ {
			genTest.Mutate(0.0067, 0.067, 0.1)
			testEval := Evaluate(genTest, tar, l2Norm)
			if testEval > genBestEval {
				genBest.CopyGenotypeFrom(genTest)
				genBestEval = testEval
			} else {
				genTest.CopyGenotypeFrom(genBest)
			}
		}
		if (it+1)%logEvery == 0 || it == 0 {
			fmt.Println("Iteration", it, "best eval", genBestEval)
			SaveImg("imgs/src.png", Vec2Img(src))
			SaveImg("imgs/tar.png", Vec2Img(tar))
			res := genBest.Generate()
			SaveImg("imgs/res.png", Vec2Img(res))
			SaveImg("imgs/mat.png", Mat2Img(genBest.matrix))
			SaveImg("imgs/vec.png", Vec2Img(genBest.vector))
		}
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
