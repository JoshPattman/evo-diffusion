package main

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"os"
	"runtime/pprof"
	"time"

	"gonum.org/v1/gonum/mat"
)

func main() {
	// Training loop params
	maxDuration := 24 * time.Hour * 3 / 2
	resetTargetEvery := 400000000
	logEvery := 100
	drawEvery := 8000
	datasetPath := "./dataset-simpler"
	doProfiling := false

	// Algorithm tunable params
	/*useSparseRegNet := false
	useDoubleDenseRegNet := true*/
	weightMutationMax := 0.0067
	weightMutationChance := 0.067
	vecMutationAmount := 0.1
	updateRate := 1.0
	decayRate := 0.2
	timesteps := 10
	numMutations := 1
	numThreads := 1
	/*// Sparse specific
	moveConProb := 0.01
	avgConnectionsPerNode := 15
	sparseWeightMutationMax := 0.01
	// Double dense specific
	doubleDenseHidden := 28
	doubleDenseUseRelu := false*/

	if doProfiling {
		maxDuration = 10 * time.Second
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

	// Create the log file
	logFile, err := os.Create("./imgs/log.csv")
	if err != nil {
		panic(err)
	}
	defer logFile.Close()
	fmt.Fprintf(logFile, "generation,best_eval\n")

	// Create the two genotypes
	currentGenotype := NewGenotype(imgVolume, vecMutationAmount, numMutations)
	currentRegNet := NewDenseRegNetwork(imgVolume, updateRate, decayRate, weightMutationMax, numMutations, weightMutationChance)

	testGenotypes := make([]*Genotype, numThreads)
	testRegNets := make([]HCRegNet, numThreads)

	for i := range testGenotypes {
		testGenotypes[i] = currentGenotype.Clone().(*Genotype)
		testRegNets[i] = currentRegNet.Clone().(HCRegNet)
	}

	startTime := time.Now()
	var tar *mat.VecDense
	var currentEval float64
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
			currentGenotype = NewGenotype(currentGenotype.Vector.Len(), currentGenotype.ValsMaxMut, currentGenotype.NumMutations)
		}

		currentEval = Evaluate(currentGenotype, currentRegNet, tar, timesteps)

		// Mutate the test genotype and evaluate it
		betterGenotypes := make([]HillClimbable, 0, len(testGenotypes))
		betterRegNets := make([]HillClimbable, 0, len(testRegNets))

		for i := range testGenotypes {
			testGenotypes[i].MutateFrom(currentGenotype)
			testRegNets[i].MutateFrom(currentGenotype)
			testEval := Evaluate(testGenotypes[i], testRegNets[i], tar, timesteps)
			if testEval > currentEval {
				betterGenotypes = append(betterGenotypes, testGenotypes[i])
				betterRegNets = append(betterRegNets, testRegNets[i])
			}
		}

		if len(betterGenotypes) > 0 {
			//currentGenotype.AverageFrom(betterGenotypes)
			//currentRegNet.AverageFrom(betterRegNets)
			currentGenotype = betterGenotypes[0].Clone().(*Genotype)
			currentRegNet = betterRegNets[0].Clone().(*DenseRegNetwork)
		}

		// Add to the log file every logEvery generations
		if gen%logEvery == 0 || gen == 1 {
			fmt.Fprintf(logFile, "%d,%.3f\n", gen, currentEval)
		}

		// Draw the images every drawEvery generations
		if gen%drawEvery == 0 || gen == 1 {
			fmt.Printf("G %v (%v): %3f\n", gen, time.Since(startTime), currentEval)
			res := currentRegNet.Run(currentGenotype.Vector, timesteps)
			SaveImg("imgs/tar.png", Vec2Img(tar))
			SaveImg("imgs/res.png", Vec2Img(res))
			SaveImg("imgs/int.png", GenerateIntermediateDiagram(currentRegNet, 20, timesteps, timesteps*3, imgSize))
		}
	}

	fmt.Println("Finished in", time.Since(startTime))
}

func GenerateIntermediateDiagram(r HCRegNet, rows, trainingTimesteps, timesteps, imgSize int) image.Image {
	imgVolume := imgSize * imgSize
	resultss := make([][]*mat.VecDense, rows)
	for i := range resultss {
		resultss[i] = r.RunWithIntermediateStates(NewGenotype(imgVolume, 0, 0).Vector, timesteps)
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
