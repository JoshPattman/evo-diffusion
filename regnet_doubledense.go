package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

var _ RegNetwork = &DenseRegNetwork{}

func NewDoubleDenseRegNetwork(nodes, hiddenNodes int, updateRate float64, decayRate float64, weightsMaxMult float64, postLoopProscessing PostLoopProscessing, performWeightClamp bool) *DoubleDenseRegNetwork {
	return &DoubleDenseRegNetwork{
		WeightsA:            mat.NewDense(hiddenNodes, nodes, nil),
		WeightsB:            mat.NewDense(nodes, hiddenNodes, nil),
		UpdateRate:          updateRate,
		DecayRate:           decayRate,
		WeightsMaxMut:       weightsMaxMult,
		PostLoopProscessing: postLoopProscessing,
		PerformWeightClamp:  performWeightClamp,
	}
}

type DoubleDenseRegNetwork struct {
	WeightsA      *mat.Dense
	WeightsB      *mat.Dense
	UpdateRate    float64
	DecayRate     float64
	WeightsMaxMut float64
	// The function to apply to the state after each timestep
	PostLoopProscessing PostLoopProscessing
	// Should we clamp the weights to -1 to 1
	PerformWeightClamp bool
}

func (d *DoubleDenseRegNetwork) Run(genotype *mat.VecDense, timesteps int) *mat.VecDense {
	state := mat.VecDenseCopyOf(genotype)
	hiddentState := mat.NewVecDense(d.WeightsA.RawMatrix().Rows, nil)
	stateUpdate := mat.NewVecDense(state.Len(), nil)
	for i := 0; i < timesteps; i++ {
		hiddentState.MulVec(d.WeightsA, state)
		stateUpdate.MulVec(d.WeightsB, hiddentState)
		ApplyAllVec(stateUpdate, tanh)
		// Decay the state
		state.ScaleVec(1-d.DecayRate, state)
		// Add the update to the state
		state.AddScaledVec(state, d.UpdateRate, stateUpdate)
		switch d.PostLoopProscessing {
		case ClampPostProcessing:
			ApplyAllVec(state, clamp(-1, 1))
		case TanhPostProcessing:
			ApplyAllVec(state, tanh)
		}
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
		stateUpdate.MulVec(d.WeightsB, hiddentState)
		ApplyAllVec(stateUpdate, tanh)
		// Decay the state
		state.ScaleVec(1-d.DecayRate, state)
		// Add the update to the state
		state.AddScaledVec(state, d.UpdateRate, stateUpdate)
		switch d.PostLoopProscessing {
		case ClampPostProcessing:
			ApplyAllVec(state, clamp(-1, 1))
		case TanhPostProcessing:
			ApplyAllVec(state, tanh)
		}
		states[i+1] = mat.VecDenseCopyOf(state)
	}
	return states
}

func (d *DoubleDenseRegNetwork) Mutate() {
	addition := d.WeightsMaxMut * (rand.Float64()*2 - 1)
	if rand.Float64() < 0.5 {
		r, c := d.WeightsA.Dims()
		ri := rand.Intn(r)
		ci := rand.Intn(c)
		newVal := d.WeightsA.At(ri, ci) + addition
		if d.PerformWeightClamp {
			newVal = clamp(-1, 1)(newVal)
		}
		d.WeightsA.Set(ri, ci, newVal)
	} else {
		r, c := d.WeightsB.Dims()
		ri := rand.Intn(r)
		ci := rand.Intn(c)
		newVal := d.WeightsB.At(ri, ci) + addition
		if d.PerformWeightClamp {
			newVal = clamp(-1, 1)(newVal)
		}
		d.WeightsB.Set(ri, ci, newVal)
	}
}

func (d *DoubleDenseRegNetwork) CopyFrom(other RegNetwork) {
	otherDense := other.(*DoubleDenseRegNetwork)
	d.WeightsA.Copy(otherDense.WeightsA)
	d.WeightsB.Copy(otherDense.WeightsB)
	d.UpdateRate = otherDense.UpdateRate
	d.DecayRate = otherDense.DecayRate
	d.WeightsMaxMut = otherDense.WeightsMaxMut
	d.PostLoopProscessing = otherDense.PostLoopProscessing
	d.PerformWeightClamp = otherDense.PerformWeightClamp
}

func (d *DoubleDenseRegNetwork) WeightsMatrix() *mat.Dense {
	// Assume we dont use relu, so we can just multiply the two matricies together
	res := mat.NewDense(d.WeightsB.RawMatrix().Rows, d.WeightsA.RawMatrix().Cols, nil)
	res.Product(d.WeightsB, d.WeightsA)
	return res
}
