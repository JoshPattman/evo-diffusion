package main

import (
	"fmt"
	"io"
	"os"
	"reflect"
)

type CSVLogger[T any] struct {
	writer io.Writer
}

func NewFileCSVLogger[T any](filename string) (*CSVLogger[T], error) {
	file, err := os.Create(filename)
	if err != nil {
		return nil, err
	}
	return NewCSVLogger[T](file), nil
}

func NewCSVLogger[T any](writer io.Writer) *CSVLogger[T] {
	c := &CSVLogger[T]{writer: writer}
	var row T
	val := reflect.ValueOf(row)
	for fi := 0; fi < val.NumField(); fi++ {
		// Write the field name to the writer
		c.writer.Write([]byte(val.Type().Field(fi).Name))
		// Write a comma
		if fi < val.NumField()-1 {
			c.writer.Write([]byte(","))
		}
	}
	// Write a newline
	c.writer.Write([]byte("\n"))
	return c
}

func (c *CSVLogger[T]) Log(row T) {
	val := reflect.ValueOf(row)
	for fi := 0; fi < val.NumField(); fi++ {
		// Write the field value to the writer
		c.writer.Write([]byte(fmt.Sprint(val.Field(fi).Interface())))
		// Write a comma
		if fi < val.NumField()-1 {
			c.writer.Write([]byte(","))
		}
	}
	// Write a newline
	c.writer.Write([]byte("\n"))
}

func (c *CSVLogger[T]) Close() error {
	if closer, ok := c.writer.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}
