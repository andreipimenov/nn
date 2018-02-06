package nn

import (
	"encoding/json"
	"errors"
	"io/ioutil"
	"math"
	"math/rand"
	"time"
)

//FFN - Feed Forward Node
type FFN struct {
	InputNeurons  int           `json:"inputNeurons"`
	HiddenNeurons int           `json:"hiddenNeurons"`
	OutputNeurons int           `json:"outputNeurons"`
	X             [][]float64   `json:"-"`
	E             [][]float64   `json:"-"`
	B             [][]float64   `json:"b"`
	W             [][][]float64 `json:"w"`
	DW            [][][]float64 `json:"dw"`
	F             []F           `json:"-"`
}

//F - Activation function
type F func(x float64) float64

//DefaultF - Default activation function
func DefaultF(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

//DefaultDF - Default derivative of activation fuction
func DefaultDF(x float64) float64 {
	return x * (1 - x)
}

//Log - Logging function
type Log func(currentEpoch int, currentRate float64)

//NewFFN - Create new Feed Forward Node
func NewFFN(inputNeurons, outputNeurons int, hiddenNeurons int, f []F) (*FFN, error) {
	if inputNeurons < 1 {
		return nil, errors.New("input neurons less than 1")
	}
	if outputNeurons < 1 {
		return nil, errors.New("output neurons less than 1")
	}
	if hiddenNeurons < 1 {
		return nil, errors.New("hidden neurons less than 1")
	}
	if f != nil && len(f) != 2 {
		return nil, errors.New("activation functions slice length is not equal to 2")
	}
	if f == nil {
		f = make([]F, 2)
	}
	if f[0] == nil || f[1] == nil {
		f[0] = DefaultF
		f[1] = DefaultDF
	}

	rand.Seed(time.Now().Unix())
	x := make([][]float64, 3)
	x[0] = make([]float64, inputNeurons)
	x[1] = make([]float64, hiddenNeurons)
	x[2] = make([]float64, outputNeurons)

	e := make([][]float64, 2)
	e[0] = make([]float64, hiddenNeurons)
	e[1] = make([]float64, outputNeurons)

	b := make([][]float64, 2)
	b[0] = make([]float64, hiddenNeurons)
	b[1] = make([]float64, outputNeurons)
	for i := 0; i < 2; i++ {
		for j := 0; j < len(b[i]); j++ {
			b[i][j] = rand.Float64() - 0.5
		}
	}

	w := make([][][]float64, 2)
	for i := 0; i < 2; i++ {
		w[i] = make([][]float64, len(x[i]))
		for j := 0; j < len(w[i]); j++ {
			w[i][j] = make([]float64, len(x[i+1]))
			for k := 0; k < len(w[i][j]); k++ {
				w[i][j][k] = rand.Float64() - 0.5
			}
		}
	}

	dw := make([][][]float64, 2)
	for i := 0; i < 2; i++ {
		dw[i] = make([][]float64, len(x[i]))
		for j := 0; j < len(dw[i]); j++ {
			dw[i][j] = make([]float64, len(x[i+1]))
		}
	}

	return &FFN{
		InputNeurons:  inputNeurons,
		HiddenNeurons: hiddenNeurons,
		OutputNeurons: outputNeurons,
		X:             x,
		E:             e,
		B:             b,
		W:             w,
		DW:            dw,
		F:             f,
	}, nil
}

//Forward - Feed input data forward
func (n *FFN) Forward(input []float64) {
	n.X[0] = input
	for layer := 1; layer < len(n.X); layer++ {
		for neuron := 0; neuron < len(n.X[layer]); neuron++ {
			prevLayer := layer - 1
			x := n.B[prevLayer][neuron]
			for prevNeuron := 0; prevNeuron < len(n.X[prevLayer]); prevNeuron++ {
				x += n.X[prevLayer][prevNeuron] * n.W[prevLayer][prevNeuron][neuron]
			}
			n.X[layer][neuron] = n.F[0](x)
		}
	}
}

//Backward - Backpropagate errors, correct weights and biases
func (n *FFN) Backward(output []float64, speed float64, moment float64) {
	for neuron := 0; neuron < len(n.X[len(n.X)-1]); neuron++ {
		x := n.X[len(n.X)-1][neuron]
		n.E[len(n.E)-1][neuron] = (output[neuron] - x) * n.F[1](x)
	}

	for layer := len(n.X) - 2; layer > 0; layer-- {
		for neuron := 0; neuron < len(n.X[layer]); neuron++ {
			e := float64(0)
			for nextNeuron := 0; nextNeuron < len(n.W[layer][neuron]); nextNeuron++ {
				e += n.W[layer][neuron][nextNeuron] * n.E[layer][nextNeuron]
			}
			x := n.X[layer][neuron]
			e *= n.F[1](x)
			n.E[layer-1][neuron] = e
		}
	}

	for layer := 0; layer < len(n.W); layer++ {
		for neuron := 0; neuron < len(n.W[layer]); neuron++ {
			for nextNeuron := 0; nextNeuron < len(n.W[layer][neuron]); nextNeuron++ {
				dw := speed*n.X[layer][neuron]*n.E[layer][nextNeuron] + moment*n.DW[layer][neuron][nextNeuron]
				n.DW[layer][neuron][nextNeuron] = dw
				n.W[layer][neuron][nextNeuron] += dw
			}
		}
	}

	for layer := 0; layer < len(n.B); layer++ {
		for nextNeuron := 0; nextNeuron < len(n.B[layer]); nextNeuron++ {
			db := speed * n.E[layer][nextNeuron]
			n.B[layer][nextNeuron] += db
		}
	}

}

//Rate - mean squared error rate function
func (n *FFN) Rate(inputs [][]float64, outputs [][]float64) float64 {
	rate := float64(0)
	for set := 0; set < len(outputs); set++ {
		e := float64(0)
		n.Forward(inputs[set])
		for neuron := 0; neuron < len(n.X[len(n.X)-1]); neuron++ {
			e += math.Pow((n.X[len(n.X)-1][neuron] - outputs[set][neuron]), 2)
		}
		rate += e
	}
	return rate / float64(len(outputs))
}

//Train - train network until error rate reached
func (n *FFN) Train(inputs [][]float64, outputs [][]float64, speed float64, moment float64, rate float64, epoch int, log Log) (float64, int, error) {
	if len(inputs) != len(outputs) {
		return 0, 0, errors.New("inputs length is not equal to outputs length")
	}
	if speed <= 0 {
		return 0, 0, errors.New("speed less or equal to 0")
	}
	if moment < 0 {
		return 0, 0, errors.New("moment less than 0")
	}
	if rate < 0 {
		return 0, 0, errors.New("rate less than 0")
	}
	if epoch <= 0 {
		return 0, 0, errors.New("epoch less or equal to 0")
	}
	var currentRate float64
	for currentEpoch := 0; ; currentEpoch++ {
		for set := 0; set < len(inputs); set++ {
			n.Forward(inputs[set])
			n.Backward(outputs[set], speed, moment)
			currentRate = n.Rate(inputs, outputs)
			if currentRate <= rate {
				return currentRate, currentEpoch, nil
			}
			if currentEpoch > epoch {
				return currentRate, currentEpoch, errors.New("maximum count of epochs exceeded")
			}
		}
		if log != nil {
			log(currentEpoch, currentRate)
		}
	}
}

//Read - feed one set of input data and read the output
func (n *FFN) Read(input []float64) []float64 {
	n.Forward(input)
	return n.X[len(n.X)-1]
}

//FromFile - Load saved network from file (note: activation function and derivation cannot being loaded)
func (n *FFN) FromFile(filename string) error {
	fileData, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}
	v := &FFN{}
	err = json.Unmarshal(fileData, &v)
	if err != nil {
		return err
	}
	n.InputNeurons = v.InputNeurons
	n.HiddenNeurons = v.HiddenNeurons
	n.OutputNeurons = v.OutputNeurons
	n.B = v.B
	n.W = v.W
	n.DW = v.DW
	return nil
}

//Dump - Save neural network to file (note: activation function and derivation cannot being saved)
func (n *FFN) Dump(filename string) error {
	fileData, _ := json.Marshal(&n)
	return ioutil.WriteFile(filename, fileData, 0644)
}
