package main

import (
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"time"
)

func main() {
	iterations := 2000000000
	subiterations := 500
	l2Norm := 0.0
	imgSize := 10
	logEvery := 2
	datasetPath := "./dataset-simple"

	imgVolume := imgSize * imgSize

	images, err := LoadDataset(datasetPath)
	if err != nil {
		panic(err)
	}
	fmt.Println("Loaded", len(images), "images")

	generation := make([]*Genotype, 10)
	for gi := range generation {
		generation[gi] = NewGenotype(imgVolume, 0.1)
	}
	startTime := time.Now()
	for it := 0; it < iterations; it++ {
		if time.Since(startTime) > 8*time.Hour {
			break
		}

		tar := images[0] //images[rand.Intn(len(images))] // //
		src := NewSrcVec(imgVolume)

		for gi := range generation {
			generation[gi].CopySrcVec(src)
		}
		itBestEval := -99.0
		for sit := 0; sit < subiterations; sit++ {
			bestEvaluation := math.Inf(-1)
			bestIndex := -1
			for gi := range generation {
				generation[gi].Mutate(0.1, 0.03, 0.3, 0.15, 0)
				evaluation := Evaluate(generation[gi], tar, l2Norm)
				if evaluation > bestEvaluation {
					bestEvaluation = evaluation
					bestIndex = gi
				}
			}
			for gi := range generation {
				if gi != bestIndex {
					generation[gi].CopyGenotypeFrom(generation[bestIndex])
				}
			}
			itBestEval = bestEvaluation
		}
		if (it+1)%logEvery == 0 || it == 0 {
			fmt.Println("Iteration", it, "best eval", itBestEval)
			genBest := generation[0]
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
