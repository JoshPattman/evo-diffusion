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

type RegNetType uint8

const (
	DenseRegNet RegNetType = iota
	DoubleDenseRegNet
	BitNetRegNet
)

func main() {
	// Training loop params
	maxDuration := 24 * time.Hour * 3 / 2
	resetTargetEvery := 4000
	logEvery := 100
	drawEvery := resetTargetEvery * 3
	datasetPath := "dataset-simpler"
	doProfiling := false

	// Algorithm tunable params
	regNetType := DoubleDenseRegNet
	weightMutationMax := 0.0067
	weightMutationChance := 0.067
	vecMutationAmount := 0.1
	updateRate := 1.0
	decayRate := 0.2
	timesteps := 10
	// Double dense specific
	doubleDenseHidden := 20
	doubleDenseUseRelu := false

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
	images, imgSizeX, imgSizeY, err := LoadDataset(datasetPath)
	if err != nil {
		panic(err)
	}
	//images = images[:2]
	imgVolume := imgSizeX * imgSizeY
	fmt.Println("Loaded", len(images), "images of size", imgSizeX, "x", imgSizeY, "(", imgVolume, "pixels )")
	fmt.Println("Min img val", mat.Min(images[0]), "Max img val", mat.Max(images[0]))

	// First compute the hebb weights
	hebbWeights := GenerateHebbWeights(images, 30, 0.02)
	// Save the hebb weights
	SaveImg("imgs/hebb.png", Mat2Img(hebbWeights, 1))
	// Save an intermediate diagram using hebb weights
	if imgSizeX == imgSizeY {
		regnet := NewDenseRegNetwork(imgVolume, updateRate, decayRate, 0)
		regnet.Weights = hebbWeights
		SaveImg("imgs/hebb_intermediate.png", GenerateIntermediateDiagram(regnet, 20, timesteps, timesteps*3, imgSizeX))
	}

	// Create the log file
	logFile, err := os.Create("./imgs/log.csv")
	if err != nil {
		panic(err)
	}
	defer logFile.Close()
	fmt.Fprintf(logFile, "generation,best_eval\n")

	// Create the two genotypes
	bestGenotype := NewGenotype(imgVolume, vecMutationAmount)
	testGenotype := NewGenotype(imgVolume, vecMutationAmount)
	var bestRegNet, testRegNet RegNetwork
	testGenotype.CopyFrom(bestGenotype)
	switch regNetType {
	case DoubleDenseRegNet:
		bestRegNet = NewDoubleDenseRegNetowrk(imgVolume, doubleDenseHidden, updateRate, decayRate, weightMutationMax, doubleDenseUseRelu)
		testRegNet = NewDoubleDenseRegNetowrk(imgVolume, doubleDenseHidden, updateRate, decayRate, weightMutationMax, doubleDenseUseRelu)
	case BitNetRegNet:
		bestRegNet = NewDenseBitNetRegNetwork(imgVolume, updateRate/100, decayRate/10)
		testRegNet = NewDenseBitNetRegNetwork(imgVolume, updateRate/100, decayRate/10)
	case DenseRegNet:
		bestRegNet = NewDenseRegNetwork(imgVolume, updateRate, decayRate, weightMutationMax)
		testRegNet = NewDenseRegNetwork(imgVolume, updateRate, decayRate, weightMutationMax)

	}

	testRegNet.CopyFrom(bestRegNet)

	startTime := time.Now()
	var tar *mat.VecDense
	bestEval := math.Inf(-1)
	// Main loop
	gen := 0
	for {
		gen++
		// Stop training if time exceeds time limit
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
			fmt.Fprintf(logFile, "%d,%.3f\n", gen, bestEval)
		}

		// Draw the images every drawEvery generations
		if gen%drawEvery == 0 || gen == 1 {
			fmt.Printf("G %v (%v): %3f\n", gen, time.Since(startTime), bestEval)
			weightMax := mat.Max(bestRegNet.WeightsMatrix())
			weightMin := mat.Min(bestRegNet.WeightsMatrix())
			if math.Abs(weightMin) > weightMax {
				weightMax = math.Abs(weightMin)
			}
			SaveImg("imgs/mat.png", Mat2Img(bestRegNet.WeightsMatrix(), weightMax))

			res := bestRegNet.Run(bestGenotype.Vector, timesteps)

			if imgSizeX == imgSizeY {
				SaveImg("imgs/tar.png", Vec2Img(tar, imgSizeX, imgSizeY))
				SaveImg("imgs/res.png", Vec2Img(res, imgSizeX, imgSizeY))
				SaveImg("imgs/vec.png", Vec2Img(bestGenotype.Vector, imgSizeX, imgSizeY))
				SaveImg("imgs/int.png", GenerateIntermediateDiagram(bestRegNet, 20, timesteps, timesteps*3, imgSizeX))
			}
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
			draw.Draw(img, image.Rect(1+icol*(imgSize+1), 1+irow*(imgSize+1), 1+(icol+1)*(imgSize+1), 1+(irow+1)*(imgSize+1)), Vec2Img(res, imgSize, imgSize), image.Point{}, draw.Src)
		}
	}
	return img
}
