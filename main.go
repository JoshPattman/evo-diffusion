package main

import (
	"fmt"
	"math"
	"os"
	"strings"
	"sync"

	"gonum.org/v1/gonum/mat"
)

type TransferFuncType uint8

const (
	Dense TransferFuncType = iota
	DoubleDense
	GroupedDense
	SparseChems
	Sparse
	BitDense
	MaskedDense
)

type Dataset string

const (
	Arbitary       Dataset = "arbitary"
	Arbitary2      Dataset = "arbitary2"
	Stalks         Dataset = "datasets/stalks"
	StalksReduced  Dataset = "datasets/stalks_reduced"
	StalksFull     Dataset = "datasets/stalks_full"
	Darwin         Dataset = "datasets/darwin"
	Simple         Dataset = "datasets/simple"
	Simpler        Dataset = "datasets/simpler"
	Modular        Dataset = "modular"
	ModularReduced Dataset = "modular_reduced"
	None           Dataset = ""
)

type ExperimentConfig struct {
	DatasetPath    Dataset
	GenDatasetPath Dataset

	TransferFuncType  TransferFuncType
	BMutationRate     float64
	MaskedProbability float64
	L2                float64

	MaxGenerations          int
	TargetSwitchGenerations int

	LogEvery             int
	FileSuffix           string
	EnableWeightLog      bool
	DrawEvery            int
	EnableDatasetDiagram bool
	EnableHebbDiagram    bool
	EnableBestiary       bool
	EnableFlatBestiary   bool
	EnableWeightDiagram  bool
	EnableDebugDiagrams  bool
	EnableDevelopDiagram bool

	EnableStdout bool
}

func main() {
	os.RemoveAll("imgs")
	os.Mkdir("imgs", os.ModePerm)

	wg := &sync.WaitGroup{}

	runtest := func(tf TransferFuncType, testName string, repNum int, enableLogging bool, maskedProbability, l2, bMutationRate float64) {
		params := ExperimentConfig{
			DatasetPath:             Stalks,
			GenDatasetPath:          StalksFull,
			MaxGenerations:          3500000,
			TargetSwitchGenerations: 25000,
			LogEvery:                100,
			DrawEvery:               12500,
			EnableBestiary:          true,
			EnableWeightDiagram:     true,
			EnableHebbDiagram:       true,

			TransferFuncType:  tf,
			FileSuffix:        fmt.Sprintf(":%s_%d", testName, repNum),
			MaskedProbability: maskedProbability,
			L2:                l2,
			EnableStdout:      enableLogging,
			BMutationRate:     bMutationRate,
		}
		test(params)
		wg.Done()
	}

	repeats := 1
	wg.Add(repeats * 1)
	for i := 0; i < repeats; i++ {
		go runtest(MaskedDense, "masked", i, true, 0.0038, 0, 0.0058)
		//go runtest(MaskedDense, "dense", i, false, 0, 0, 0.0067)
		//go runtest(MaskedDense, "masked", i, false, 0.0038, 0, 0.0058)
		//go runtest(MaskedDense, "dense+l2", i, false, 0, 5.0, 0.0067)
		//go runtest(MaskedDense, "masked+l2", i, i == 0, 0.0038, 5.0, 0.0058)
	}

	wg.Wait()
}

func test(config ExperimentConfig) {
	if config.LogEvery == 0 {
		panic("remember to set LogEvery to a non-zero value to log the results")
	}
	// Load the dataset
	images, imgSizeX, imgSizeY, err := LoadDataset(string(config.DatasetPath))
	if err != nil {
		panic(err)
	}
	// Load the test dataset if applicable
	var testImages []*mat.VecDense
	if config.GenDatasetPath != None {
		tis, tx, ty, err := LoadDataset(string(config.GenDatasetPath))
		if err != nil {
			panic(err)
		}
		if tx != imgSizeX || ty != imgSizeY {
			panic("Test dataset size mismatch")
		}
		testImages = tis
	} else {
		testImages = images
	}
	// Calulate the image volume
	imgVolume := imgSizeX * imgSizeY
	//fmt.Println("Loaded", len(images), "images of size", imgSizeX, "x", imgSizeY, "(", imgVolume, "pixels )")
	//fmt.Println("Min img val", mat.Min(images[0]), "Max img val", mat.Max(images[0]))

	// Regulatory network params
	vecMutationAmount := 0.1
	numVecMutations := 1
	updateRate := 1.0
	decayRate := 0.2
	timesteps := 10

	// Generate a dataset diagram
	if config.EnableDatasetDiagram {
		if imgSizeX != imgSizeY {
			panic("cannot generate dataset diagram for non-square images")
		}
		dxf := math.Sqrt(float64(len(images)))
		// round up
		dx := int(dxf)
		if dxf-float64(dx) > 0 {
			dx++
		}
		SaveImg("imgs/dataset"+config.FileSuffix+".png", GenerateDatasetDiagram(images, dx, dx, imgSizeX))
	}

	// Generate hebb weights
	if config.EnableHebbDiagram {
		// First compute the hebb weights
		hebbWeights := GenerateHebbWeights(images)
		// Save the hebb weights
		SaveImg("imgs/hebb_weights_max1"+config.FileSuffix+".png", Mat2Img(hebbWeights, 1))
		// Save an intermediate diagram using hebb weights
		if imgSizeX == imgSizeY {
			tf := NewDenseTransferFunc(imgVolume, true, 0)
			tf.Weights = hebbWeights
			regnet := NewRegulatoryNetwork(tf, updateRate, decayRate, timesteps, imgVolume)
			SaveImg("imgs/hebb_bestiary"+config.FileSuffix+".png", GenerateBestiaryDiagram(regnet, 10, 10, imgSizeX))
		}
	}

	// Create the log file
	type EvoLogRow struct {
		Generation                   int
		BestEval                     float64
		FlatWeights                  string
		PercentUniqueClassesProduced float64
		PercentOfProductionsValid    float64
	}
	csvLogger, err := NewFileCSVLogger[EvoLogRow]("./imgs/log" + config.FileSuffix + ".csv")
	if err != nil {
		panic(err)
	}
	defer csvLogger.Close()

	// Create the two genotypes
	bestGenotype := NewGenotype(imgVolume, vecMutationAmount)
	testGenotype := NewGenotype(imgVolume, vecMutationAmount)
	testGenotype.CopyFrom(bestGenotype)

	bestRegNet := NewRegulatoryNetwork(createTransferFunc(config.TransferFuncType, imgVolume, config), updateRate, decayRate, timesteps, imgVolume)
	testRegNet := NewRegulatoryNetwork(createTransferFunc(config.TransferFuncType, imgVolume, config), updateRate, decayRate, timesteps, imgVolume)
	testRegNet.CopyFrom(bestRegNet)

	var tar *mat.VecDense
	bestEval := math.Inf(-1)

	// Main loop
	for gen, pulse := range GenerationalLoadingBar(config.MaxGenerations, config.TargetSwitchGenerations, 15, &bestEval, config.EnableStdout) {
		// Reset the target image every TargetSwitchGenerations generations (and src vector too)
		if pulse {
			tar = images[(gen/config.TargetSwitchGenerations)%len(images)]
			bestGenotype = NewGenotype(bestGenotype.Vector.Len(), bestGenotype.ValsMaxMut)
			bestEval = Evaluate(bestGenotype, bestRegNet, tar, config.L2)
		}

		// Mutate the test genotype and evaluate it
		for range numVecMutations {
			testGenotype.Mutate()
		}
		testRegNet.Mutate()

		testEval := Evaluate(testGenotype, testRegNet, tar, config.L2)
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
		if gen%config.LogEvery == 0 || gen == 1 {
			var flatWeights []string
			if config.EnableWeightLog {
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
			produceNum := len(testImages) * 4
			numUniqueClassesSeen, numInTargets := GenerateAndCountUnique(bestRegNet, produceNum, testImages)
			csvLogger.Log(EvoLogRow{
				Generation:                   gen,
				BestEval:                     bestEval,
				FlatWeights:                  strings.Join(flatWeights, ":"),
				PercentUniqueClassesProduced: float64(numUniqueClassesSeen) / float64(len(testImages)),
				PercentOfProductionsValid:    float64(numInTargets) / float64(produceNum),
			})
		}

		// Draw the diagrams sometimes
		if config.DrawEvery > 0 && (gen%config.DrawEvery == 0 || gen == 1) {
			if config.EnableWeightDiagram {
				wm := bestRegNet.WeightsMatrix()
				weightMax := mat.Max(wm)
				weightMin := mat.Min(wm)
				if math.Abs(weightMin) > weightMax {
					weightMax = math.Abs(weightMin)
				}
				SaveImg("imgs/evo_weights"+config.FileSuffix+".png", Mat2Img(wm, weightMax))
				SaveImg("imgs/evo_weights_max1"+config.FileSuffix+".png", Mat2Img(wm, 1))
			}
			if config.EnableDebugDiagrams {
				res := bestRegNet.Run(bestGenotype.Vector)
				SaveImg("imgs/evo_target"+config.FileSuffix+".png", Vec2Img(tar, imgSizeX, imgSizeY))
				SaveImg("imgs/evo_result"+config.FileSuffix+".png", Vec2Img(res, imgSizeX, imgSizeY))
				SaveImg("imgs/evo_input"+config.FileSuffix+".png", Vec2Img(bestGenotype.Vector, imgSizeX, imgSizeY))
			}
			if config.EnableBestiary {
				SaveImg("imgs/evo_bestiary"+config.FileSuffix+".png", GenerateBestiaryDiagram(bestRegNet, 10, 10, imgSizeX))
			}
			if config.EnableFlatBestiary {
				SaveImg("imgs/evo_fig12e"+config.FileSuffix+".png", GenerateFig12EDiagram(bestRegNet, 30, imgVolume))
			}
		}
	}

	if config.EnableDevelopDiagram {
		type GraphDRow struct {
			Id       int
			Timestep int
			Vals     string
		}
		dcsvLogger, err := NewFileCSVLogger[GraphDRow]("./imgs/d" + config.FileSuffix + ".csv")
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
}

func createTransferFunc(transferFuncType TransferFuncType, genotypeSize int, config ExperimentConfig) TransferFunc {
	// Create transfer func
	switch transferFuncType {
	case Dense:
		return NewDenseTransferFunc(genotypeSize, true, config.BMutationRate)
	case DoubleDense:
		return NewDoubleDenseTransferFunc(genotypeSize, genotypeSize/4, true, config.BMutationRate)
	case Sparse:
		return NewSparseTransferFunc(genotypeSize, 10, 0.01)
	case BitDense:
		return NewBitDenseTransferFunc(genotypeSize, 8, true, 0.003, 0.0025)
	case MaskedDense:
		return NewMaskedDenseTransferFunc(genotypeSize, true, config.BMutationRate, config.MaskedProbability)
	}
	panic("invalid transfer function type")
}
