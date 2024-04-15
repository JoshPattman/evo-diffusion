package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

var _ TransferFunc = &SparseTransferFunc{}

type SparseConnection struct {
	From   int
	Weight float64
}

type SparseTransferFunc struct {
	Connections [][]SparseConnection
	MutStd      float64
}

func NewSparseTransferFunc(genotypeSize, numSparseConnections int, mutStd float64) *SparseTransferFunc {
	conns := make([][]SparseConnection, genotypeSize)
	for i := range conns {
		conns[i] = make([]SparseConnection, numSparseConnections)
		for j := 0; j < numSparseConnections; j++ {
			conns[i][j].From = rand.Intn(genotypeSize)
			conns[i][j].Weight = rand.NormFloat64() * mutStd
		}
	}
	return &SparseTransferFunc{
		Connections: conns,
		MutStd:      mutStd,
	}
}

func (t *SparseTransferFunc) Mutate() {
	conns := t.Connections[rand.Intn(len(t.Connections))]
	conn := conns[rand.Intn(len(conns))]
	conn.Weight += t.MutStd * rand.NormFloat64()
}

func (t *SparseTransferFunc) CopyFrom(other TransferFunc) {
	otherT := other.(*SparseTransferFunc)
	for i := range t.Connections {
		copy(t.Connections[i], otherT.Connections[i])
	}
}

func (t *SparseTransferFunc) Transfer(state, into *mat.VecDense) {
	for i := range t.Connections {
		sum := 0.0
		for _, conn := range t.Connections[i] {
			sum += conn.Weight * state.AtVec(conn.From)
		}
		into.SetVec(i, sum)
	}
}

func (t *SparseTransferFunc) GenotypeSize() int {
	return len(t.Connections)
}

func (t *SparseTransferFunc) CombinedWeightsMatrix() *mat.Dense {
	gs := t.GenotypeSize()
	empty := mat.NewDense(gs, gs, nil)
	for i := range t.Connections {
		for _, conn := range t.Connections[i] {
			empty.Set(i, conn.From, conn.Weight+empty.At(i, conn.From))
		}
	}
	return empty
}
