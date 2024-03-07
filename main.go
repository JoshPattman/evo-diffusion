package main

import (
	"encoding/gob"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math"
	"os"
	"time"

	"gonum.org/v1/gonum/mat"
)

func main() {
	// Algorithm tunable params
	maxDuration := 8 * time.Hour
	resetTargetEvery := 2000
	logEvery := 100
	drawEvery := resetTargetEvery * 15
	l2Norm := 0.0
	datasetPath := "./dataset-simple"

	// Load the dataset
	images, imgSize, err := LoadDataset(datasetPath)
	if err != nil {
		panic(err)
	}
	//images = images[:2]
	imgVolume := imgSize * imgSize
	fmt.Println("Loaded", len(images), "images")

	// Create the two genotypes
	genBest := NewGenotype(imgVolume)
	genTest := NewGenotype(imgVolume)
	genTest.CopyGenotypeFrom(genBest)

	// Create the log file
	logFile, err := os.Create("./imgs/log.csv")
	if err != nil {
		panic(err)
	}
	defer logFile.Close()
	fmt.Fprintf(logFile, "generation,best_eval\n")

	startTime := time.Now()
	var tar *mat.VecDense
	var src *mat.VecDense
	bestEval := math.Inf(-1)
	// Main loop
	gen := 0
	for {
		gen++
		// Stop training if time exceeds 5 minutes
		if time.Since(startTime) > maxDuration {
			break
		}

		// Reset the target image every resetTargetEvery generations (and src vector too)
		if (gen-1)%resetTargetEvery == 0 {
			tar = images[(gen/resetTargetEvery)%len(images)]
			src = NewSrcVec(imgVolume)
			genTest.CopySrcVec(src)
			genBest.CopySrcVec(src)
			bestEval = math.Inf(-1)
		}

		// Mutate the test genotype and evaluate it
		genTest.Mutate(0.0067, 0.067, 0.1)
		testEval := Evaluate(genTest, tar, l2Norm)
		// If the test genotype is better than the best genotype, copy it over, otherwise reset to prev best
		if testEval > bestEval {
			genBest.CopyGenotypeFrom(genTest)
			bestEval = testEval
		} else {
			genTest.CopyGenotypeFrom(genBest)
		}

		// Add to the log file every logEvery generations
		if gen%logEvery == 0 || gen == 1 {
			fmt.Fprintf(logFile, "%d,%.3f\n", gen, bestEval)
		}

		// Draw the images every drawEvery generations
		if gen%drawEvery == 0 || gen == 1 {
			fmt.Printf("G %v (%v): %3f\n", gen, time.Since(startTime), bestEval)
			res := genBest.Generate()
			SaveImg("imgs/src.png", Vec2Img(src))
			SaveImg("imgs/tar.png", Vec2Img(tar))
			SaveImg("imgs/res.png", Vec2Img(res))
			SaveImg("imgs/mat.png", Mat2Img(genBest.Matrix))
			SaveImg("imgs/vec.png", Vec2Img(genBest.Vector))
			SaveImg("imgs/int.png", GenerateIntermediateDiagram(genBest, 10, imgSize))
			// gob encode the genotype
			func() {
				f, err := os.Create("imgs/gen.gob")
				if err != nil {
					panic(err)
				}
				defer f.Close()
				enc := gob.NewEncoder(f)
				err = enc.Encode(genBest)
				if err != nil {
					panic(err)
				}
			}()
		}
	}

	fmt.Println("Finished in", time.Since(startTime))

	SaveImg("imgs/intermediate.png", GenerateIntermediateDiagram(genBest, 50, imgSize))
}

func GenerateIntermediateDiagram(g *Genotype, rows, imgSize int) image.Image {
	imgVolume := imgSize * imgSize
	resultss := make([][]*mat.VecDense, rows)
	for i := range resultss {
		src := NewSrcVec(imgVolume)
		g.CopySrcVec(src)
		resultss[i] = g.GenerateWithIntermediate()
	}

	img := image.NewRGBA(image.Rect(0, 0, 1+(imgSize+1)*(g.Iterations+1), 1+(imgSize+1)*rows))
	draw.Draw(img, img.Bounds(), &image.Uniform{color.RGBA{100, 100, 0, 255}}, image.Point{}, draw.Src)
	for irow, results := range resultss {
		for icol, res := range results {
			draw.Draw(img, image.Rect(1+icol*(imgSize+1), 1+irow*(imgSize+1), 1+(icol+1)*(imgSize+1), 1+(irow+1)*(imgSize+1)), Vec2Img(res), image.Point{}, draw.Src)
		}
	}
	return img
}
