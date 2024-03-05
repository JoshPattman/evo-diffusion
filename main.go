package main

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"time"

	"gonum.org/v1/gonum/mat"
)

func main() {
	iterations := 10000000
	subiterations := 2000
	l2Norm := 0.0
	imgSize := 10
	logEvery := 5
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
		if time.Since(startTime) > 5*time.Minute {
			break
		}

		//tar := images[rand.Intn(len(images))]
		tar := images[it%2]
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

	fmt.Println("Finished in", time.Since(startTime))

	{
		rows := 10
		resultss := make([][]*mat.VecDense, rows)
		for i := range resultss {
			src := NewSrcVec(imgVolume)
			genBest.CopySrcVec(src)
			resultss[i] = genBest.GenerateWithIntermediate()
		}

		img := image.NewRGBA(image.Rect(0, 0, 1+(imgSize+1)*(genBest.iterations+1), 1+(imgSize+1)*rows))
		draw.Draw(img, img.Bounds(), &image.Uniform{color.RGBA{100, 100, 0, 255}}, image.Point{}, draw.Src)
		for irow, results := range resultss {
			for icol, res := range results {
				draw.Draw(img, image.Rect(1+icol*(imgSize+1), 1+irow*(imgSize+1), 1+(icol+1)*(imgSize+1), 1+(irow+1)*(imgSize+1)), Vec2Img(res), image.Point{}, draw.Src)
			}
		}
		SaveImg("imgs/intermediate.png", img)
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
