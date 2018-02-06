package main

import (
	"fmt"

	"github.com/andreipimenov/nn"
)

func main() {

	//Define optional logging function. It receives current epoch and current rate every training step.
	log := func(currentEpoch int, currentRate float64) {
		fmt.Printf("Current epoch: %d. Current Rate: %.03f\n", currentEpoch, currentRate)
	}

	//Init neural network with 2 input neurons, 1 output neuron and 4 hidden neurons (1 hidden layer) and using default activation function
	n, _ := nn.NewFFN(2, 1, 4, nil)

	//Define inputs
	//For example, will use XOR form boolean algebra
	inputs := [][]float64{
		[]float64{0, 0},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{1, 1},
	}

	//And outputs
	outputs := [][]float64{
		[]float64{0},
		[]float64{1},
		[]float64{1},
		[]float64{0},
	}

	//Train our network with
	//0.2 - training speed
	//0.05 - momentum value (helps to overcome local minimums)
	//0.01 - allowed rate value
	//10000 - maximum value of training epochs
	rate, epoch, err := n.Train(inputs, outputs, 0.2, 0.05, 0.01, 10000, log)
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
		return
	}
	fmt.Printf("Rate: %.03f\nEpoch: %d\n", rate, epoch)

	fmt.Println("Testing")
	fmt.Printf("Input: %v. Expected output: %v. Output: %v\n", inputs[0], outputs[0], n.Read(inputs[0]))
	fmt.Printf("Input: %v. Expected output: %v. Output: %v\n", inputs[1], outputs[1], n.Read(inputs[1]))
	fmt.Printf("Input: %v. Expected output: %v. Output: %v\n", inputs[2], outputs[2], n.Read(inputs[2]))
	fmt.Printf("Input: %v. Expected output: %v. Output: %v\n", inputs[3], outputs[3], n.Read(inputs[3]))
}
