package main

import "gonum.org/v1/gonum/mat"

func GenerateHebbWeights(targets []*mat.VecDense) *mat.Dense {
	weights := mat.NewDense(targets[0].Len(), targets[0].Len(), nil)
	weightsUpdate := mat.NewDense(targets[0].Len(), targets[0].Len(), nil)
	for _, target := range targets {
		// inputs * targets, but both inputs and targets are the same vector
		weightsUpdate.Outer(1, target, target)
		weights.Add(weights, weightsUpdate)
	}
	// We can just do this at the end as it does not affect the weigth updates
	weights.Apply(func(i, j int, v float64) float64 {
		return clamp(-1, 1)(v)
	}, weights)
	return weights
}
