package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

var _ TransferFunc = &BitDenseTransferFunc{}

type BitDenseTransferFunc struct {
	WeightsA           *mat.Dense
	WeightsB           *mat.Dense
	PerformWeightClamp bool
	WeightsMaxMut      float64
	MutateBitChance    float64
	NumChemicals       int
	GenotypeLength     int
	chemBuf            *mat.VecDense
}

func NewBitDenseTransferFunc(genotypeLength, numChemicals int, performWeightClamp bool, weightsMaxMut, mutateBitChance float64) *BitDenseTransferFunc {
	randLayerA := make([]float64, genotypeLength*numChemicals)
	for i := range randLayerA {
		if rand.Float64() < 0.2 {
			if rand.Float64() < 0.5 {
				randLayerA[i] = -1
			} else {
				randLayerA[i] = 1
			}
		}
	}
	return &BitDenseTransferFunc{
		WeightsA:           mat.NewDense(numChemicals, genotypeLength, randLayerA),
		WeightsB:           mat.NewDense(genotypeLength, numChemicals, nil),
		chemBuf:            mat.NewVecDense(numChemicals, nil),
		PerformWeightClamp: performWeightClamp,
		WeightsMaxMut:      weightsMaxMut,
		MutateBitChance:    mutateBitChance,
		NumChemicals:       numChemicals,
		GenotypeLength:     genotypeLength,
	}
}

func (d *BitDenseTransferFunc) Mutate() {
	if rand.Float64() < d.MutateBitChance {
		m := d.WeightsA
		r, c := m.Dims()
		ri, ci := rand.Intn(r), rand.Intn(c)
		v := rand.Float64()
		newVal := 0.0
		if v < 0.33 {
			newVal = -1
		} else if v < 0.66 {
			newVal = 0
		} else {
			newVal = 1
		}
		m.Set(ri, ci, newVal)
	} else {
		m := d.WeightsB
		r, c := m.Dims()
		ri, ci := rand.Intn(r), rand.Intn(c)
		addition := d.WeightsMaxMut * (rand.Float64()*2 - 1)
		newVal := m.At(ri, ci) + addition
		if d.PerformWeightClamp {
			newVal = clamp(-1, 1)(newVal)
		}
		m.Set(ri, ci, newVal)
	}

}

func (d *BitDenseTransferFunc) CopyFrom(other TransferFunc) {
	d.WeightsA.Copy(other.(*BitDenseTransferFunc).WeightsA)
	d.WeightsB.Copy(other.(*BitDenseTransferFunc).WeightsB)
}

func (d *BitDenseTransferFunc) Transfer(state, into *mat.VecDense) {
	d.chemBuf.MulVec(d.WeightsA, state)
	into.MulVec(d.WeightsB, d.chemBuf)
}

func (d *BitDenseTransferFunc) CombinedWeightsMatrix() *mat.Dense {
	res := mat.NewDense(d.GenotypeLength, d.GenotypeLength, nil)
	res.Mul(d.WeightsB, d.WeightsA)
	return res
}
