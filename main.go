package main

import (
	"fmt"
	"math"
	"math/rand"
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
	datasetPath := "./dataset-simple"
	timesteps := 10

	// Load the dataset
	images, imgSize, err := LoadDataset(datasetPath)
	if err != nil {
		panic(err)
	}
	images = images[:2]
	imgVolume := imgSize * imgSize
	fmt.Println("Loaded", len(images), "images")

	// Create the two genotypes
	bestGenotype := NewGenotype(imgVolume, 0.1)
	testGenotype := NewGenotype(imgVolume, 0.1)
	testGenotype.CopyFrom(bestGenotype)
	bestDenseReg := NewDenseRegNetwork(imgVolume, 1, 0.2, 0.067)
	testDenseReg := NewDenseRegNetwork(imgVolume, 1, 0.2, 0.067)
	testDenseReg.CopyFrom(bestDenseReg)

	// Create the log file
	logFile, err := os.Create("./imgs/log.csv")
	if err != nil {
		panic(err)
	}
	defer logFile.Close()
	fmt.Fprintf(logFile, "generation,best_eval\n")

	startTime := time.Now()
	var tar *mat.VecDense
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
			bestGenotype = NewGenotype(bestGenotype.Vector.Len(), bestGenotype.ValsMaxMut)
			bestEval = Evaluate(bestGenotype, bestDenseReg, tar, timesteps)
		}

		// Mutate the test genotype and evaluate it
		testGenotype.Mutate()
		if rand.Float64() < 0.067 {
			testDenseReg.Mutate()
		}

		testEval := Evaluate(testGenotype, testDenseReg, tar, timesteps)
		// If the test genotype is better than the best genotype, copy it over, otherwise reset to prev best
		if testEval > bestEval {
			bestGenotype.CopyFrom(testGenotype)
			bestDenseReg.CopyFrom(testDenseReg)
			bestEval = testEval
		} else {
			testGenotype.CopyFrom(bestGenotype)
			testDenseReg.CopyFrom(bestDenseReg)
		}

		// Add to the log file every logEvery generations
		if gen%logEvery == 0 || gen == 1 {
			fmt.Fprintf(logFile, "%d,%.3f\n", gen, bestEval)
		}

		// Draw the images every drawEvery generations
		if gen%drawEvery == 0 || gen == 1 {
			fmt.Printf("G %v (%v): %3f\n", gen, time.Since(startTime), bestEval)
			res := bestDenseReg.Run(bestGenotype.Vector, timesteps)
			SaveImg("imgs/tar.png", Vec2Img(tar))
			SaveImg("imgs/res.png", Vec2Img(res))
			SaveImg("imgs/mat.png", Mat2Img(bestDenseReg.Weights))
			SaveImg("imgs/vec.png", Vec2Img(bestGenotype.Vector))
		}
	}

	fmt.Println("Finished in", time.Since(startTime))
}
