package main

import (
	"math/rand"
	"slices"

	"gonum.org/v1/gonum/mat"
)

type PixelGroup struct {
	Coords     []int
	DataLength int
}

func NewPixelGroup(dataLength, groupSize int) *PixelGroup {
	coords := make([]int, groupSize)
	for i := 0; i < groupSize; i++ {
		coords[i] = rand.Intn(dataLength)
	}
	return &PixelGroup{Coords: coords, DataLength: dataLength}
}

func (p *PixelGroup) Sum(data *mat.VecDense) float64 {
	sum := 0.0
	for _, coord := range p.Coords {
		sum += data.AtVec(coord)
	}
	return sum / float64(len(p.Coords))
}

func (p *PixelGroup) Mutate() {
	p.Coords[rand.Intn(len(p.Coords))] = rand.Intn(p.DataLength)
}

type RegNetGrouped struct {
	Groups             []*PixelGroup
	Weights            *mat.Dense
	UpdateRate         float64
	DecayRate          float64
	WeightsMaxMut      float64
	GroupsMutChance    float64
	PerformWeightClamp bool
}

func NewGroupedDenseRegNet(nodes, groups, groupSize int, updateRate float64, decayRate float64, weightsMaxMut, groupsMutChance float64, performWeightClamp bool) *RegNetGrouped {
	groupsSlice := make([]*PixelGroup, groups)
	for i := 0; i < groups; i++ {
		groupsSlice[i] = NewPixelGroup(nodes, groupSize)
	}
	weights := mat.NewDense(nodes, groups, nil)
	return &RegNetGrouped{
		Groups:             groupsSlice,
		Weights:            weights,
		UpdateRate:         updateRate,
		DecayRate:          decayRate,
		WeightsMaxMut:      weightsMaxMut,
		GroupsMutChance:    groupsMutChance,
		PerformWeightClamp: performWeightClamp,
	}
}

func (g *RegNetGrouped) CopyFrom(other RegNetwork) {
	g2 := other.(*RegNetGrouped)
	for i, group := range g2.Groups {
		g.Groups[i].Coords = slices.Clone(group.Coords)
	}
	g.Weights.Copy(g2.Weights)
	g.UpdateRate = g2.UpdateRate
	g.DecayRate = g2.DecayRate
	g.WeightsMaxMut = g2.WeightsMaxMut
	g.PerformWeightClamp = g2.PerformWeightClamp
}

func (g *RegNetGrouped) Mutate() {
	if rand.Float64() < g.GroupsMutChance {
		g.Groups[rand.Intn(len(g.Groups))].Mutate()
	} else {
		rs, cs := g.Weights.Dims()
		ri, ci := rand.Intn(rs), rand.Intn(cs)
		newVal := g.Weights.At(ri, ci) + (rand.Float64()*2-1)*g.WeightsMaxMut
		if g.PerformWeightClamp {
			newVal = clamp(-1, 1)(newVal)
		}
		g.Weights.Set(ri, ci, newVal)
	}
}

func (g *RegNetGrouped) Run(genotype *mat.VecDense, timesteps int) (finalState *mat.VecDense) {
	state := mat.VecDenseCopyOf(genotype)
	groupsState := mat.NewVecDense(g.Weights.RawMatrix().Cols, nil)
	stateUpdate := mat.NewVecDense(state.Len(), nil)
	for i := 0; i < timesteps; i++ {
		for gi := range g.Groups {
			groupsState.SetVec(gi, g.Groups[gi].Sum(state))
		}
		stateUpdate.MulVec(g.Weights, groupsState)
		ApplyAllVec(stateUpdate, tanh)
		state.ScaleVec(1-g.DecayRate, state)
		state.AddScaledVec(state, g.UpdateRate, stateUpdate)
	}
	return state
}

func (g *RegNetGrouped) RunWithIntermediateStates(genotype *mat.VecDense, timesteps int) (states []*mat.VecDense) {
	states = make([]*mat.VecDense, timesteps+1)
	state := mat.VecDenseCopyOf(genotype)
	groupsState := mat.NewVecDense(g.Weights.RawMatrix().Cols, nil)
	stateUpdate := mat.NewVecDense(state.Len(), nil)
	for i := 0; i < timesteps; i++ {
		states[i] = mat.VecDenseCopyOf(state)
		for gi := range g.Groups {
			groupsState.SetVec(gi, g.Groups[gi].Sum(state))
		}
		stateUpdate.MulVec(g.Weights, groupsState)
		ApplyAllVec(stateUpdate, tanh)
		state.ScaleVec(1-g.DecayRate, state)
		state.AddScaledVec(state, g.UpdateRate, stateUpdate)
	}
	states[timesteps] = mat.VecDenseCopyOf(state)
	return states
}

func (g *RegNetGrouped) WeightsMatrix() *mat.Dense {
	return g.Weights
}
