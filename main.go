package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type RegNetType uint8

const (
	DenseRegNet RegNetType = iota
	DoubleDenseRegNet
	BitNetRegNet
)

func main() {
	// Training loop params
	maxGenerations := 1000000
	resetTargetEvery := 4000
	logEvery := 100
	drawEvery := resetTargetEvery * 3
	datasetPath := "dataset-darwin"

	// Algorithm tunable params
	regNetType := DoubleDenseRegNet
	weightMutationMax := 0.0067
	weightMutationChance := 1.0 //0.067
	vecMutationAmount := 0.1
	updateRate := 1.0
	decayRate := 0.2
	timesteps := 10
	// Double dense specific
	doubleDenseHidden := 20

	// Load the dataset
	images, imgSizeX, imgSizeY, err := LoadDataset(datasetPath)
	if err != nil {
		panic(err)
	}
	imgVolume := imgSizeX * imgSizeY
	fmt.Println("Loaded", len(images), "images of size", imgSizeX, "x", imgSizeY, "(", imgVolume, "pixels )")
	fmt.Println("Min img val", mat.Min(images[0]), "Max img val", mat.Max(images[0]))

	// First compute the hebb weights
	hebbWeights := GenerateHebbWeights(images, 30, 0.02)
	// Save the hebb weights
	SaveImg("imgs/hebb_weights.png", Mat2Img(hebbWeights, 1))
	// Save an intermediate diagram using hebb weights
	if imgSizeX == imgSizeY {
		regnet := NewDenseRegNetwork(imgVolume, updateRate, decayRate, 0)
		regnet.Weights = hebbWeights
		SaveImg("imgs/hebb_intermediate.png", GenerateIntermediateDiagram(regnet, 20, timesteps, timesteps*3, imgSizeX))
	}

	// Create the log file
	type EvoLogRow struct {
		Generation int
		BestEval   float64
	}
	csvLogger, err := NewFileCSVLogger[EvoLogRow]("./imgs/log.csv")
	if err != nil {
		panic(err)
	}
	defer csvLogger.Close()

	// Create the two genotypes
	bestGenotype := NewGenotype(imgVolume, vecMutationAmount)
	testGenotype := NewGenotype(imgVolume, vecMutationAmount)
	var bestRegNet, testRegNet RegNetwork
	testGenotype.CopyFrom(bestGenotype)
	switch regNetType {
	case DoubleDenseRegNet:
		bestRegNet = NewDoubleDenseRegNetwork(imgVolume, doubleDenseHidden, updateRate, decayRate, weightMutationMax)
		testRegNet = NewDoubleDenseRegNetwork(imgVolume, doubleDenseHidden, updateRate, decayRate, weightMutationMax)
	case BitNetRegNet:
		bestRegNet = NewDenseBitNetRegNetwork(imgVolume, updateRate/100, decayRate/10)
		testRegNet = NewDenseBitNetRegNetwork(imgVolume, updateRate/100, decayRate/10)
	case DenseRegNet:
		bestRegNet = NewDenseRegNetwork(imgVolume, updateRate, decayRate, weightMutationMax)
		testRegNet = NewDenseRegNetwork(imgVolume, updateRate, decayRate, weightMutationMax)
	}

	testRegNet.CopyFrom(bestRegNet)

	var tar *mat.VecDense
	bestEval := math.Inf(-1)
	// Main loop

	for gen := 1; gen <= maxGenerations; gen++ {
		// Reset the target image every resetTargetEvery generations (and src vector too)
		if (gen-1)%resetTargetEvery == 0 {
			tar = images[(gen/resetTargetEvery)%len(images)]
			bestGenotype = NewGenotype(bestGenotype.Vector.Len(), bestGenotype.ValsMaxMut)
			bestEval = Evaluate(bestGenotype, bestRegNet, tar, timesteps)
		}

		// Mutate the test genotype and evaluate it
		for i := 0; i < rand.Intn(5)+1; i++ {
			testGenotype.Mutate()
			if rand.Float64() < weightMutationChance {
				testRegNet.Mutate()
			}
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
			csvLogger.Log(EvoLogRow{Generation: gen, BestEval: bestEval})
		}

		// Draw the images every drawEvery generations
		if gen%drawEvery == 0 || gen == 1 {
			fmt.Printf("G %v: %3f\n", gen, bestEval)
			weightMax := mat.Max(bestRegNet.WeightsMatrix())
			weightMin := mat.Min(bestRegNet.WeightsMatrix())
			if math.Abs(weightMin) > weightMax {
				weightMax = math.Abs(weightMin)
			}
			SaveImg("imgs/evo_weights.png", Mat2Img(bestRegNet.WeightsMatrix(), weightMax))

			res := bestRegNet.Run(bestGenotype.Vector, timesteps)

			if imgSizeX == imgSizeY {
				SaveImg("imgs/evo_target.png", Vec2Img(tar, imgSizeX, imgSizeY))
				SaveImg("imgs/evo_result.png", Vec2Img(res, imgSizeX, imgSizeY))
				SaveImg("imgs/evo_input.png", Vec2Img(bestGenotype.Vector, imgSizeX, imgSizeY))
				SaveImg("imgs/evo_intermediate.png", GenerateIntermediateDiagram(bestRegNet, 20, timesteps, timesteps*3, imgSizeX))
			}
		}
	}
}
