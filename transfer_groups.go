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

type GroupedTransferFunc struct {
	Groups             []*PixelGroup
	Weights            *mat.Dense
	WeightsMaxMut      float64
	GroupsMutChance    float64
	PerformWeightClamp bool
}

func NewGroupedTransferFunc(nodes, groups, groupSize int, weightsMaxMut, groupsMutChance float64, performWeightClamp bool) *GroupedTransferFunc {
	groupsSlice := make([]*PixelGroup, groups)
	for i := 0; i < groups; i++ {
		groupsSlice[i] = NewPixelGroup(nodes, groupSize)
	}
	weights := mat.NewDense(nodes, groups, nil)
	return &GroupedTransferFunc{
		Groups:             groupsSlice,
		Weights:            weights,
		WeightsMaxMut:      weightsMaxMut,
		GroupsMutChance:    groupsMutChance,
		PerformWeightClamp: performWeightClamp,
	}
}

func (g *GroupedTransferFunc) CopyFrom(other TransferFunc) {
	g2 := other.(*GroupedTransferFunc)
	for i, group := range g2.Groups {
		g.Groups[i].Coords = slices.Clone(group.Coords)
	}
	g.Weights.Copy(g2.Weights)
	g.WeightsMaxMut = g2.WeightsMaxMut
	g.PerformWeightClamp = g2.PerformWeightClamp
}

func (g *GroupedTransferFunc) Mutate() {
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

func (g *GroupedTransferFunc) CombinedWeightsMatrix() *mat.Dense {
	return g.Weights
}

func (g *GroupedTransferFunc) Transfer(state, into *mat.VecDense) {
	buf := mat.NewVecDense(len(g.Groups), nil)
	for i, group := range g.Groups {
		buf.SetVec(i, group.Sum(state))
	}
	into.MulVec(g.Weights, buf)
}
