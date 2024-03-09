package main

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math"
	"math/rand"
	"os"
	"runtime/pprof"
	"time"

	"gonum.org/v1/gonum/mat"
)

func main() {
	// Training loop params
	maxDuration := 10 * time.Hour
	resetTargetEvery := 2000
	logEvery := 100
	drawEvery := resetTargetEvery * 45
	datasetPath := "./dataset-simple"
	doProfiling := false

	// Algorithm tunable params
	useSparseRegNet := true
	weightMutationMax := 0.0067
	weightMutationChance := 0.067
	vecMutationAmount := 0.1
	updateRate := 1.0
	decayRate := 0.2
	timesteps := 10
	// Sparse specific
	moveConProb := 0.02
	avgConnectionsPerNode := 10
	sparseWeightMutationMax := 0.0017

	if doProfiling {
		pf, err := os.Create("cpu.prof")
		if err != nil {
			panic(err)
		}
		defer pf.Close()
		pprof.StartCPUProfile(pf)
		defer pprof.StopCPUProfile()
		defer fmt.Println("Stopped profiling")
	}

	// Load the dataset
	images, imgSize, err := LoadDataset(datasetPath)
	if err != nil {
		panic(err)
	}
	//images = images[:2]
	imgVolume := imgSize * imgSize
	fmt.Println("Loaded", len(images), "images")

	// Create the two genotypes
	bestGenotype := NewGenotype(imgVolume, vecMutationAmount)
	testGenotype := NewGenotype(imgVolume, vecMutationAmount)
	var bestRegNet, testRegNet RegNetwork
	testGenotype.CopyFrom(bestGenotype)
	if useSparseRegNet {
		// As there are fewer rates, reduce the chance of mutation
		//weightMutationChance *= float64(avgConnectionsPerNode) / float64(imgVolume)
		bestRegNet = NewSparseRegNetwork(imgVolume, imgVolume*avgConnectionsPerNode, updateRate, decayRate, sparseWeightMutationMax, moveConProb)
		testRegNet = NewSparseRegNetwork(imgVolume, imgVolume*avgConnectionsPerNode, updateRate, decayRate, sparseWeightMutationMax, moveConProb)
	} else {
		bestRegNet = NewDenseRegNetwork(imgVolume, updateRate, decayRate, weightMutationMax)
		testRegNet = NewDenseRegNetwork(imgVolume, updateRate, decayRate, weightMutationMax)
	}

	testRegNet.CopyFrom(bestRegNet)

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
			fmt.Println("Time exceeded", maxDuration, "stopping after generation", gen)
			break
		}

		// Reset the target image every resetTargetEvery generations (and src vector too)
		if (gen-1)%resetTargetEvery == 0 {
			tar = images[(gen/resetTargetEvery)%len(images)]
			bestGenotype = NewGenotype(bestGenotype.Vector.Len(), bestGenotype.ValsMaxMut)
			bestEval = Evaluate(bestGenotype, bestRegNet, tar, timesteps)
		}

		// Mutate the test genotype and evaluate it
		testGenotype.Mutate()
		if rand.Float64() < weightMutationChance {
			testRegNet.Mutate()
		}

		testEval := Evaluate(testGenotype, testRegNet, tar, timesteps)
		// If the test genotype is better than the best genotype, copy it over, otherwise reset to prev best
		if testEval > bestEval {
			bestGenotype.CopyFrom(testGenotype)
			bestRegNet.CopyFrom(testRegNet)
			bestEval = testEval
		} else {
			testGenotype.CopyFrom(bestGenotype)
			testRegNet.CopyFrom(bestRegNet)
		}

		// Add to the log file every logEvery generations
		if gen%logEvery == 0 || gen == 1 {
			fmt.Fprintf(logFile, "%d,%.3f\n", gen, bestEval)
		}

		// Draw the images every drawEvery generations
		if gen%drawEvery == 0 || gen == 1 {
			fmt.Printf("G %v (%v): %3f\n", gen, time.Since(startTime), bestEval)
			res := bestRegNet.Run(bestGenotype.Vector, timesteps)
			SaveImg("imgs/tar.png", Vec2Img(tar))
			SaveImg("imgs/res.png", Vec2Img(res))
			SaveImg("imgs/mat.png", Mat2Img(bestRegNet.WeightsMatrix()))
			SaveImg("imgs/vec.png", Vec2Img(bestGenotype.Vector))
			SaveImg("imgs/int.png", GenerateIntermediateDiagram(bestRegNet, 20, timesteps, timesteps*3, imgSize))
		}
	}

	fmt.Println("Finished in", time.Since(startTime))
}

func GenerateIntermediateDiagram(r RegNetwork, rows, trainingTimesteps, timesteps, imgSize int) image.Image {
	imgVolume := imgSize * imgSize
	resultss := make([][]*mat.VecDense, rows)
	for i := range resultss {
		resultss[i] = r.RunWithIntermediateStates(NewGenotype(imgVolume, 0).Vector, timesteps)
	}

	img := image.NewRGBA(image.Rect(0, 0, 1+(imgSize+1)*(timesteps+1), 1+(imgSize+1)*rows))
	draw.Draw(img, image.Rect(0, 0, imgSize*trainingTimesteps, img.Bounds().Max.Y), &image.Uniform{color.RGBA{0, 255, 0, 255}}, image.Point{}, draw.Src)
	draw.Draw(img, image.Rect(imgSize*trainingTimesteps, 0, img.Bounds().Max.X, img.Bounds().Max.Y), &image.Uniform{color.RGBA{0, 0, 255, 255}}, image.Point{}, draw.Src)
	for irow, results := range resultss {
		for icol, res := range results {
			draw.Draw(img, image.Rect(1+icol*(imgSize+1), 1+irow*(imgSize+1), 1+(icol+1)*(imgSize+1), 1+(irow+1)*(imgSize+1)), Vec2Img(res), image.Point{}, draw.Src)
		}
	}
	return img
}
