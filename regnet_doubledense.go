package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

var _ RegNetwork = &DenseRegNetwork{}

func NewDoubleDenseRegNetowrk(nodes, hiddenNodes int, updateRate float64, decayRate float64, weightsMaxMult float64, useHiddenRelu bool) *DoubleDenseRegNetwork {
	return &DoubleDenseRegNetwork{
		WeightsA:      mat.NewDense(hiddenNodes, nodes, nil),
		WeightsB:      mat.NewDense(nodes, hiddenNodes, nil),
		UpdateRate:    updateRate,
		DecayRate:     decayRate,
		WeightsMaxMut: weightsMaxMult,
		UseHiddenRelu: useHiddenRelu,
	}
}

type DoubleDenseRegNetwork struct {
	WeightsA      *mat.Dense
	WeightsB      *mat.Dense
	UpdateRate    float64
	DecayRate     float64
	WeightsMaxMut float64
	UseHiddenRelu bool
}

func (d *DoubleDenseRegNetwork) Run(genotype *mat.VecDense, timesteps int) *mat.VecDense {
	state := mat.VecDenseCopyOf(genotype)
	hiddentState := mat.NewVecDense(d.WeightsA.RawMatrix().Rows, nil)
	stateUpdate := mat.NewVecDense(state.Len(), nil)
	for i := 0; i < timesteps; i++ {
		hiddentState.MulVec(d.WeightsA, state)
		if d.UseHiddenRelu {
			ApplyAllVec(hiddentState, relu)
		}
		stateUpdate.MulVec(d.WeightsB, hiddentState)
		ApplyAllVec(stateUpdate, tanh)
		// Decay the state
		state.ScaleVec(1-d.DecayRate, state)
		// Add the update to the state
		state.AddScaledVec(state, d.UpdateRate, stateUpdate)
		// Ensure the state is still in range -1 to 1
		ApplyAllVec(state, clamp(0, 1))
	}
	return state
}

func (d *DoubleDenseRegNetwork) RunWithIntermediateStates(genotype *mat.VecDense, timesteps int) []*mat.VecDense {
	states := make([]*mat.VecDense, timesteps+1)
	state := mat.VecDenseCopyOf(genotype)
	states[0] = mat.VecDenseCopyOf(state)
	hiddentState := mat.NewVecDense(d.WeightsA.RawMatrix().Rows, nil)
	stateUpdate := mat.NewVecDense(state.Len(), nil)
	for i := 0; i < timesteps; i++ {
		hiddentState.MulVec(d.WeightsA, state)
		if d.UseHiddenRelu {
			ApplyAllVec(hiddentState, relu)
		}
		stateUpdate.MulVec(d.WeightsB, hiddentState)
		ApplyAllVec(stateUpdate, tanh)
		// Decay the state
		state.ScaleVec(1-d.DecayRate, state)
		// Add the update to the state
		state.AddScaledVec(state, d.UpdateRate, stateUpdate)
		// Ensure the state is still in range -1 to 1
		ApplyAllVec(state, clamp(0, 1))
		states[i+1] = mat.VecDenseCopyOf(state)
	}
	return states
}

func (d *DoubleDenseRegNetwork) Mutate() {
	addition := d.WeightsMaxMut * (rand.Float64()*2 - 1)
	//addition := d.WeightsMaxMut * rand.NormFloat64()
	if rand.Float64() < 0.5 {
		r, c := d.WeightsA.Dims()
		ri := rand.Intn(r)
		ci := rand.Intn(c)
		d.WeightsA.Set(ri, ci, d.WeightsA.At(ri, ci)+addition)
	} else {
		r, c := d.WeightsB.Dims()
		ri := rand.Intn(r)
		ci := rand.Intn(c)
		d.WeightsB.Set(ri, ci, d.WeightsB.At(ri, ci)+addition)
	}
}

func (d *DoubleDenseRegNetwork) CopyFrom(other RegNetwork) {
	otherDense := other.(*DoubleDenseRegNetwork)
	d.WeightsA.Copy(otherDense.WeightsA)
	d.WeightsB.Copy(otherDense.WeightsB)
	d.UpdateRate = otherDense.UpdateRate
	d.DecayRate = otherDense.DecayRate
	d.WeightsMaxMut = otherDense.WeightsMaxMut
}

func (d *DoubleDenseRegNetwork) WeightsMatrix() *mat.Dense {
	// Assume we dont use relu, so we can just multiply the two matricies together
	res := mat.NewDense(d.WeightsB.RawMatrix().Rows, d.WeightsA.RawMatrix().Cols, nil)
	res.Product(d.WeightsB, d.WeightsA)
	return res
}
