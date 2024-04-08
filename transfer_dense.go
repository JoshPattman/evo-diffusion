package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

var _ TransferFunc = &DenseTransferFunc{}

type DenseTransferFunc struct {
	Weights            *mat.Dense
	PerformWeightClamp bool
	WeightsMaxMut      float64
}

func NewDenseTransferFunc(genotypeLength int, performWeightClamp bool, weightsMaxMut float64) *DenseTransferFunc {
	return &DenseTransferFunc{
		Weights:            mat.NewDense(genotypeLength, genotypeLength, nil),
		PerformWeightClamp: performWeightClamp,
		WeightsMaxMut:      weightsMaxMut,
	}
}

func (d *DenseTransferFunc) Mutate() {
	r, c := d.Weights.Dims()
	ri := rand.Intn(r)
	ci := rand.Intn(c)
	addition := d.WeightsMaxMut * (rand.Float64()*2 - 1)
	newVal := d.Weights.At(ri, ci) + addition
	if d.PerformWeightClamp {
		newVal = clamp(-1, 1)(newVal)
	}
	d.Weights.Set(ri, ci, newVal)
}

func (d *DenseTransferFunc) CopyFrom(other TransferFunc) {
	d.Weights.Copy(other.(*DenseTransferFunc).Weights)
}

func (d *DenseTransferFunc) Transfer(state, into *mat.VecDense) {
	into.MulVec(d.Weights, state)
}

func (d *DenseTransferFunc) CombinedWeightsMatrix() *mat.Dense {
	return d.Weights
}
