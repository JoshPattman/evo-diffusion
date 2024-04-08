package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

var _ TransferFunc = &DoubleDenseTransferFunc{}

type DoubleDenseTransferFunc struct {
	WeightsA           *mat.Dense
	WeightsB           *mat.Dense
	PerformWeightClamp bool
	WeightsMaxMut      float64
	NumChemicals       int
	GenotypeLength     int
	chemBuf            *mat.VecDense
}

func NewDoubleDenseTransferFunc(genotypeLength, numChemicals int, performWeightClamp bool, weightsMaxMut float64) *DoubleDenseTransferFunc {
	randLayerA := make([]float64, genotypeLength*numChemicals)
	for i := range randLayerA {
		randLayerA[i] = (rand.Float64()*2 - 1) * weightsMaxMut
	}
	return &DoubleDenseTransferFunc{
		WeightsA:           mat.NewDense(numChemicals, genotypeLength, randLayerA),
		WeightsB:           mat.NewDense(genotypeLength, numChemicals, nil),
		chemBuf:            mat.NewVecDense(numChemicals, nil),
		PerformWeightClamp: performWeightClamp,
		WeightsMaxMut:      weightsMaxMut,
		NumChemicals:       numChemicals,
		GenotypeLength:     genotypeLength,
	}
}

func (d *DoubleDenseTransferFunc) Mutate() {
	var m *mat.Dense
	if rand.Float64() < 0.5 {
		m = d.WeightsA
	} else {
		m = d.WeightsB
	}
	r, c := m.Dims()
	ri, ci := rand.Intn(r), rand.Intn(c)
	addition := d.WeightsMaxMut * (rand.Float64()*2 - 1)
	newVal := m.At(ri, ci) + addition
	if d.PerformWeightClamp {
		newVal = clamp(-1, 1)(newVal)
	}
	m.Set(ri, ci, newVal)
}

func (d *DoubleDenseTransferFunc) CopyFrom(other TransferFunc) {
	d.WeightsA.Copy(other.(*DoubleDenseTransferFunc).WeightsA)
	d.WeightsB.Copy(other.(*DoubleDenseTransferFunc).WeightsB)
}

func (d *DoubleDenseTransferFunc) Transfer(state, into *mat.VecDense) {
	d.chemBuf.MulVec(d.WeightsA, state)
	into.MulVec(d.WeightsB, d.chemBuf)
}

func (d *DoubleDenseTransferFunc) CombinedWeightsMatrix() *mat.Dense {
	res := mat.NewDense(d.GenotypeLength, d.GenotypeLength, nil)
	res.Mul(d.WeightsB, d.WeightsA)
	return res
}
