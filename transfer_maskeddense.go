package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

var _ TransferFunc = &MaskedDenseTransferFunc{}

type MaskedDenseTransferFunc struct {
	Weights            *mat.Dense
	Mask               *mat.Dense
	mulBuf             *mat.Dense
	PerformWeightClamp bool
	WeightsMaxMut      float64
	MutateMaskChance   float64
}

func NewMaskedDenseTransferFunc(genotypeLength int, performWeightClamp bool, weightsMaxMut, mutateMaskChance float64) *MaskedDenseTransferFunc {
	ones := make([]float64, genotypeLength*genotypeLength)
	for i := range ones {
		ones[i] = 1
	}
	return &MaskedDenseTransferFunc{
		Weights:            mat.NewDense(genotypeLength, genotypeLength, nil),
		Mask:               mat.NewDense(genotypeLength, genotypeLength, ones),
		mulBuf:             mat.NewDense(genotypeLength, genotypeLength, nil),
		PerformWeightClamp: performWeightClamp,
		WeightsMaxMut:      weightsMaxMut,
		MutateMaskChance:   mutateMaskChance,
	}
}

func (d *MaskedDenseTransferFunc) Mutate() {
	r, c := d.Weights.Dims()
	ri := rand.Intn(r)
	ci := rand.Intn(c)
	if rand.Float64() < d.MutateMaskChance {
		newVal := float64(rand.Intn(3)) - 1
		d.Mask.Set(ri, ci, newVal)
	} else {
		addition := d.WeightsMaxMut * (rand.Float64()*2 - 1)
		newVal := d.Weights.At(ri, ci) + addition
		if d.PerformWeightClamp {
			newVal = clamp(-1, 1)(newVal)
		}
		d.Weights.Set(ri, ci, newVal)
	}
	d.mulBuf.MulElem(d.Weights, d.Mask)
}

func (d *MaskedDenseTransferFunc) CopyFrom(other TransferFunc) {
	d.Weights.Copy(other.(*MaskedDenseTransferFunc).Weights)
	d.Mask.Copy(other.(*MaskedDenseTransferFunc).Mask)
	d.mulBuf.Copy(other.(*MaskedDenseTransferFunc).mulBuf)
}

func (d *MaskedDenseTransferFunc) Transfer(state, into *mat.VecDense) {
	into.MulVec(d.mulBuf, state)
}

func (d *MaskedDenseTransferFunc) CombinedWeightsMatrix() *mat.Dense {
	return d.mulBuf
}
