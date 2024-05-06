package main

import (
	"fmt"
	"iter"
	"strings"
	"time"
)

func GenerationalLoadingBar(generations, pulseEvery, length int, score *float64, enabled bool) iter.Seq2[int, bool] {
	return func(yield func(int, bool) bool) {
		if enabled {
			fmt.Printf("G: %s\tS: %s", loadingBar(0, generations, length), loadingBar(0, pulseEvery, length))
		}
		lastPrint := time.Now()
		loopStart := time.Now()
		for gen := 0; gen < generations; gen++ {
			pulse := gen%pulseEvery == 0
			if !yield(gen, pulse) {
				break
			}
			if time.Since(lastPrint) > time.Second/10 || gen == generations-1 {
				seconds := time.Since(loopStart).Seconds()
				secondsLeft := seconds / float64(gen+1) * float64(generations-gen-1)
				if enabled {
					fmt.Printf("\rG: %s\tS: %s\t%.3f\t%.2fs/%.2fs", loadingBar(gen, generations, length), loadingBar(gen%pulseEvery, pulseEvery, length), *score, seconds, secondsLeft)
				}
				lastPrint = time.Now()
			}
		}
		fmt.Println()
	}
}

func loadingBar(n, maxN, length int) string {
	i := int(float64(n) / float64(maxN) * float64(length))
	return fmt.Sprintf("|%s>%s|", strings.Repeat("=", i), strings.Repeat(" ", length-i-1))
}
