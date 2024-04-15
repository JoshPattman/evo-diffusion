package main

import (
	"fmt"
	"math"
	"os"
	"strings"

	"gonum.org/v1/gonum/mat"
)

type TransferFuncType uint8

const (
	Dense TransferFuncType = iota
	DoubleDense
	GroupedDense
	SparseChems
	Sparse
)

type Dataset string

const (
	Arbitary  Dataset = "arbitary"
	Arbitary2 Dataset = "arbitary2"
	Stalks    Dataset = "datasets/stalks"
	Darwin    Dataset = "datasets/darwin"
	Simple    Dataset = "datasets/simple"
	Simpler   Dataset = "datasets/simpler"
)

func main() {
	// Training loop params
	maxGenerations := 16000000
	resetTargetEvery := 8000
	logEvery := 100
	drawEvery := resetTargetEvery * 3
	datasetPath := Simpler
	logWeights := datasetPath == Arbitary || datasetPath == Arbitary2

	// Load the dataset
	images, imgSizeX, imgSizeY, err := LoadDataset(string(datasetPath))
	if err != nil {
		panic(err)
	}
	imgVolume := imgSizeX * imgSizeY
	fmt.Println("Loaded", len(images), "images of size", imgSizeX, "x", imgSizeY, "(", imgVolume, "pixels )")
	fmt.Println("Min img val", mat.Min(images[0]), "Max img val", mat.Max(images[0]))

	// Regulatory network params
	vecMutationAmount := 0.1
	updateRate := 1.0
	decayRate := 0.2
	timesteps := 10
	transferFuncType := Dense

	// Create transfer func
	var makeTF func() TransferFunc
	switch transferFuncType {
	case Dense:
		makeTF = func() TransferFunc { return NewDenseTransferFunc(imgVolume, true, 0.0067) }
	case DoubleDense:
		makeTF = func() TransferFunc { return NewDoubleDenseTransferFunc(imgVolume, 10, true, 0.0067) }
	case Sparse:
		makeTF = func() TransferFunc { return NewSparseTransferFunc(imgVolume, 10, 0.01) }
	}

	// Clear the imgs folder
	//os.RemoveAll("imgs")
	os.Mkdir("imgs", os.ModePerm)

	// First compute the hebb weights
	hebbWeights := GenerateHebbWeights(images, 30, 0.02)
	// Save the hebb weights
	SaveImg("imgs/hebb_weights_max1.png", Mat2Img(hebbWeights, 1))
	// Save an intermediate diagram using hebb weights
	if imgSizeX == imgSizeY {
		tf := NewDenseTransferFunc(imgVolume, true, 0)
		tf.Weights = hebbWeights
		regnet := NewRegulatoryNetwork(tf, updateRate, decayRate, timesteps, imgVolume)
		SaveImg("imgs/hebb_bestiary.png", GenerateBestiaryDiagram(regnet, 10, 10, imgSizeX))
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
	testGenotype.CopyFrom(bestGenotype)

	bestRegNet := NewRegulatoryNetwork(makeTF(), updateRate, decayRate, timesteps, imgVolume)
	testRegNet := NewRegulatoryNetwork(makeTF(), updateRate, decayRate, timesteps, imgVolume)
	testRegNet.CopyFrom(bestRegNet)

	var tar *mat.VecDense
	bestEval := math.Inf(-1)

	// Main loop
	for gen := 1; gen <= maxGenerations; gen++ {
		// Reset the target image every resetTargetEvery generations (and src vector too)
		if (gen-1)%resetTargetEvery == 0 {
			tar = images[(gen/resetTargetEvery)%len(images)]
			bestGenotype = NewGenotype(bestGenotype.Vector.Len(), bestGenotype.ValsMaxMut)
			bestEval = Evaluate(bestGenotype, bestRegNet, tar)
		}

		// Mutate the test genotype and evaluate it
		testGenotype.Mutate()
		testRegNet.Mutate()

		testEval := Evaluate(testGenotype, testRegNet, tar)
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
			wm := bestRegNet.WeightsMatrix()
			weightMax := mat.Max(wm)
			weightMin := mat.Min(wm)
			if math.Abs(weightMin) > weightMax {
				weightMax = math.Abs(weightMin)
			}
			//fmt.Println("Weight max", weightMax, "Weight min", weightMin)
			SaveImg("imgs/evo_weights.png", Mat2Img(wm, weightMax))
			SaveImg("imgs/evo_weights_max1.png", Mat2Img(wm, 1))

			res := bestRegNet.Run(bestGenotype.Vector)

			SaveImg("imgs/evo_target.png", Vec2Img(tar, imgSizeX, imgSizeY))
			SaveImg("imgs/evo_result.png", Vec2Img(res, imgSizeX, imgSizeY))
			SaveImg("imgs/evo_input.png", Vec2Img(bestGenotype.Vector, imgSizeX, imgSizeY))

			if imgSizeX == imgSizeY {
				SaveImg("imgs/evo_bestiary.png", GenerateBestiaryDiagram(bestRegNet, 10, 10, imgSizeX))
			}
			if datasetPath == "arbitary" || datasetPath == "arbitary2" {
				SaveImg("imgs/evo_fig12e.png", GenerateFig12EDiagram(bestRegNet, 30, imgVolume))
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
		ints := bestRegNet.RunWithIntermediateStates(gt.Vector)
		for t, v := range ints {
			vals := make([]string, v.Len())
			for i := 0; i < v.Len(); i++ {
				vals[i] = fmt.Sprintf("%f", v.AtVec(i))
			}
			dcsvLogger.Log(GraphDRow{Id: id, Timestep: t, Vals: strings.Join(vals, ":")})
		}
	}
}
