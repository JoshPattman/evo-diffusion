package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"

	"gonum.org/v1/gonum/mat"
)

type RegNetType uint8

const (
	DenseRegNet RegNetType = iota
	DoubleDenseRegNet
)

func main() {
	// Training loop params
	maxGenerations := 80000
	resetTargetEvery := 2000
	logEvery := 100
	drawEvery := resetTargetEvery * 3
	datasetPath := "arbitary2"
	logWeights := datasetPath == "arbitary" || datasetPath == "arbitary2"

	// Algorithm tunable params
	regNetType := DenseRegNet
	weightMutationMax := 0.0067
	weightMutationChance := 1.0 //0.067 //
	vecMutationAmount := 0.1
	updateRate := 1.0
	decayRate := 0.2
	timesteps := 10
	postLoopProcessing := NoPostProcessing
	performWeightClamp := true
	// Double dense specific
	doubleDenseHidden := 40

	// Load the dataset
	images, imgSizeX, imgSizeY, err := LoadDataset(datasetPath)
	if err != nil {
		panic(err)
	}
	imgVolume := imgSizeX * imgSizeY
	fmt.Println("Loaded", len(images), "images of size", imgSizeX, "x", imgSizeY, "(", imgVolume, "pixels )")
	fmt.Println("Min img val", mat.Min(images[0]), "Max img val", mat.Max(images[0]))

	// Clear the imgs folder
	os.RemoveAll("imgs")
	os.Mkdir("imgs", os.ModePerm)

	// First compute the hebb weights
	hebbWeights := GenerateHebbWeights(images, 30, 0.02)
	// Save the hebb weights
	SaveImg("imgs/hebb_weights_max1.png", Mat2Img(hebbWeights, 1))
	// Save an intermediate diagram using hebb weights
	if imgSizeX == imgSizeY {
		regnet := NewDenseRegNetwork(imgVolume, updateRate, decayRate, 0, postLoopProcessing, performWeightClamp)
		regnet.Weights = hebbWeights
		SaveImg("imgs/hebb_intermediate.png", GenerateIntermediateDiagram(regnet, 20, timesteps, timesteps*3, imgSizeX))
	}

	// Create the log file
	type EvoLogRow struct {
		Generation  int
		BestEval    float64
		FlatWeights string
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
	case DenseRegNet:
		bestRegNet = NewDenseRegNetwork(imgVolume, updateRate, decayRate, weightMutationMax, postLoopProcessing, performWeightClamp)
		testRegNet = NewDenseRegNetwork(imgVolume, updateRate, decayRate, weightMutationMax, postLoopProcessing, performWeightClamp)
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
			var flatWeights []string
			if logWeights {
				wm := bestRegNet.WeightsMatrix()
				r, c := wm.Dims()
				flatWeights = make([]string, 0, r*c)
				for i := 0; i < r; i++ {
					for j := 0; j < c; j++ {
						flatWeights = append(flatWeights, fmt.Sprintf("%.4f", wm.At(i, j)))
					}
				}
			} else {
				flatWeights = []string{"0.0"}
			}
			csvLogger.Log(EvoLogRow{Generation: gen, BestEval: bestEval, FlatWeights: strings.Join(flatWeights, ":")})
		}

		// Draw the images every drawEvery generations
		if gen%drawEvery == 0 || gen == 1 {
			fmt.Printf("G %v: %3f\n", gen, bestEval)
			weightMax := mat.Max(bestRegNet.WeightsMatrix())
			weightMin := mat.Min(bestRegNet.WeightsMatrix())
			if math.Abs(weightMin) > weightMax {
				weightMax = math.Abs(weightMin)
			}
			//fmt.Println("Weight max", weightMax, "Weight min", weightMin)
			SaveImg("imgs/evo_weights.png", Mat2Img(bestRegNet.WeightsMatrix(), weightMax))
			SaveImg("imgs/evo_weights_max1.png", Mat2Img(bestRegNet.WeightsMatrix(), 1))

			res := bestRegNet.Run(bestGenotype.Vector, timesteps)

			SaveImg("imgs/evo_target.png", Vec2Img(tar, imgSizeX, imgSizeY))
			SaveImg("imgs/evo_result.png", Vec2Img(res, imgSizeX, imgSizeY))
			SaveImg("imgs/evo_input.png", Vec2Img(bestGenotype.Vector, imgSizeX, imgSizeY))

			if imgSizeX == imgSizeY {
				SaveImg("imgs/evo_intermediate.png", GenerateIntermediateDiagram(bestRegNet, 20, timesteps, timesteps*3, imgSizeX))
			}
			if datasetPath == "arbitary" || datasetPath == "arbitary2" {
				SaveImg("imgs/evo_fig12e.png", GenerateFig12EDiagram(bestRegNet, 30, timesteps, imgVolume))
			}
		}
	}

	type GraphDRow struct {
		Id       int
		Timestep int
		Vals     string
	}
	dcsvLogger, err := NewFileCSVLogger[GraphDRow]("./imgs/d.csv")
	if err != nil {
		panic(err)
	}
	defer dcsvLogger.Close()
	for id := 0; id < 4; id++ {
		gt := NewGenotype(imgVolume, vecMutationAmount)
		ints := bestRegNet.RunWithIntermediateStates(gt.Vector, timesteps*2)
		for t, v := range ints {
			vals := make([]string, v.Len())
			for i := 0; i < v.Len(); i++ {
				vals[i] = fmt.Sprintf("%f", v.AtVec(i))
			}
			dcsvLogger.Log(GraphDRow{Id: id, Timestep: t, Vals: strings.Join(vals, ":")})
		}
	}
}
