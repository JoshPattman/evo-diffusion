package main

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math"
	"os"
	"runtime/pprof"
	"time"

	"github.com/JoshPattman/goevo"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// Training loop params
	maxDuration := 10 * time.Hour
	resetTargetEvery := 1000
	logEvery := 100
	drawEvery := 20
	datasetPath := "./dataset-simple"
	doProfiling := false

	// Algorithm tunable params
	weightMutationMax := 0.0067
	weightMutationChance := 0.067
	vecMutationAmount := 0.15
	updateRate := 1.0
	decayRate := 0.2
	timesteps := 10
	tournamentSize := 5
	popSize := 100

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

	pop := goevo.NewSimplePopulation(func() *GenoRegPair {
		return &GenoRegPair{
			Genotype:   NewGenotype(imgVolume, vecMutationAmount),
			RegNetwork: NewDenseRegNetwork(imgVolume, updateRate, decayRate, weightMutationMax),
		}
	}, popSize)

	selection := &goevo.TournamentSelection[*GenoRegPair]{
		TournamentSize: tournamentSize,
	}

	reproduction := &GRReproduction{
		RegNetMutateChance: weightMutationChance,
	}

	startTime := time.Now()
	var tar *mat.VecDense
	// Main loop
	gen := 0
	for {
		gen++
		// Stop training if time exceeds 5 minutes
		if time.Since(startTime) > maxDuration {
			fmt.Println("Time exceeded", maxDuration, "stopping after generation", gen)
			break
		}

		// Reset both the target image and the genotypes sometimes
		if (gen-1)%resetTargetEvery == 0 {
			tar = images[(gen/resetTargetEvery)%len(images)]
			example := pop.Agents()[0].Genotype.Genotype
			newGenotype := NewGenotype(example.Vector.Len(), example.ValsMaxMut)
			agents := pop.Agents()
			for _, a := range agents {
				a.Genotype.Genotype = newGenotype.Clone()
			}
		}

		// Evaluate fitness of whole generation
		agents := pop.Agents()
		bestEval := math.Inf(-1)
		var bestGenotype *GenoRegPair
		for _, a := range agents {
			a.Fitness = Evaluate(a.Genotype.Genotype, a.Genotype.RegNetwork, tar, timesteps)
			if a.Fitness > bestEval {
				bestEval = a.Fitness
				bestGenotype = a.Genotype
			}
		}

		pop = pop.NextGeneration(selection, reproduction)

		// Add to the log file every logEvery generations
		if gen%logEvery == 0 || gen == 1 {
			fmt.Fprintf(logFile, "%d,%.3f\n", gen, bestEval)
		}

		// Draw the images every drawEvery generations
		if gen%drawEvery == 0 || gen == 1 {
			fmt.Printf("G %v (%v): %3f\n", gen, time.Since(startTime), bestEval)
			res := bestGenotype.RegNetwork.Run(bestGenotype.Genotype.Vector, timesteps)
			SaveImg("imgs/tar.png", Vec2Img(tar))
			SaveImg("imgs/res.png", Vec2Img(res))
			SaveImg("imgs/mat.png", Mat2Img(bestGenotype.RegNetwork.WeightsMatrix()))
			SaveImg("imgs/vec.png", Vec2Img(bestGenotype.Genotype.Vector))
			SaveImg("imgs/int.png", GenerateIntermediateDiagram(bestGenotype.RegNetwork, 20, timesteps, timesteps*3, imgSize))
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
