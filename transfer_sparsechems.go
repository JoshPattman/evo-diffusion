package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

var _ TransferFunc = &SparseChemTransferFunc{}

type ChemicalWeight struct {
	Chemical int
	Weight   float64
}

type SparseChemTransferFunc struct {
	Chemicals            int
	chemBuf              []float64
	Production           [][]ChemicalWeight
	Consumption          [][]ChemicalWeight
	MutateAmount         float64
	ChangeChemicalChance float64
}

func NewSparseChemTransferFunc(genotypeLength, numChemicals, numChemicalsPerNode int, weightsMaxMut float64) *SparseChemTransferFunc {
	prod := make([][]ChemicalWeight, genotypeLength)
	for i := range prod {
		prod[i] = make([]ChemicalWeight, numChemicalsPerNode)
		for j := range prod[i] {
			prod[i][j] = ChemicalWeight{
				Chemical: rand.Intn(numChemicals),
				Weight:   0, //(rand.Float64()*2 - 1) * weightsMaxMut,
			}
		}
	}
	con := make([][]ChemicalWeight, genotypeLength)
	for i := range con {
		con[i] = make([]ChemicalWeight, numChemicalsPerNode)
		for j := range con[i] {
			con[i][j] = ChemicalWeight{
				Chemical: rand.Intn(numChemicals),
				Weight:   0, //(rand.Float64()*2 - 1) * weightsMaxMut,
			}
		}
	}

	return &SparseChemTransferFunc{
		Chemicals:            numChemicals,
		chemBuf:              make([]float64, numChemicals),
		Production:           prod,
		Consumption:          con,
		MutateAmount:         weightsMaxMut,
		ChangeChemicalChance: 0.02,
	}
}

// CombinedWeightsMatrix implements TransferFunc.
func (s *SparseChemTransferFunc) CombinedWeightsMatrix() *mat.Dense {
	return mat.NewDense(10, 10, nil)
}

// CopyFrom implements TransferFunc.
func (s *SparseChemTransferFunc) CopyFrom(other TransferFunc) {
	otherSparse := other.(*SparseChemTransferFunc)
	s.Chemicals = otherSparse.Chemicals
	s.MutateAmount = otherSparse.MutateAmount
	s.ChangeChemicalChance = otherSparse.ChangeChemicalChance
	s.Production = make([][]ChemicalWeight, len(otherSparse.Production))
	for i := range otherSparse.Production {
		s.Production[i] = make([]ChemicalWeight, len(otherSparse.Production[i]))
		copy(s.Production[i], otherSparse.Production[i])
	}
	s.Consumption = make([][]ChemicalWeight, len(otherSparse.Consumption))
	for i := range otherSparse.Consumption {
		s.Consumption[i] = make([]ChemicalWeight, len(otherSparse.Consumption[i]))
		copy(s.Consumption[i], otherSparse.Consumption[i])
	}
}

// Mutate implements TransferFunc.
func (s *SparseChemTransferFunc) Mutate() {
	var cws [][]ChemicalWeight
	if rand.Float64() < 0.5 {
		cws = s.Production
	} else {
		cws = s.Consumption
	}
	i := rand.Intn(len(cws))
	j := rand.Intn(len(cws[i]))
	if rand.Float64() < s.ChangeChemicalChance {
		cws[i][j].Chemical = rand.Intn(s.Chemicals)
	} else {
		cws[i][j].Weight += (rand.Float64()*2 - 1) * s.MutateAmount
	}
}

// Transfer implements TransferFunc.
func (s *SparseChemTransferFunc) Transfer(state *mat.VecDense, into *mat.VecDense) {
	for i := 0; i < len(s.chemBuf); i++ {
		s.chemBuf[i] = 0
	}
	for i := 0; i < state.Len(); i++ {
		for ci := range s.Production[i] {
			s.chemBuf[s.Production[i][ci].Chemical] += s.Production[i][ci].Weight * state.AtVec(i)
		}
	}
	/*for i := 0; i < len(s.chemBuf); i++ {
		s.chemBuf[i] /= float64(s.Chemicals)
	}*/
	for i := 0; i < into.Len(); i++ {
		sum := 0.0
		for ci := range s.Consumption[i] {
			sum += s.Consumption[i][ci].Weight * s.chemBuf[s.Consumption[i][ci].Chemical]
		}
		into.SetVec(i, sum)
	}
}
