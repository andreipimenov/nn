package nn

import (
	"testing"
)

func TestNewFFN(t *testing.T) {
	tests := []struct {
		InputNeurons  int
		OutputNeurons int
		HiddenNeurons int
		ExpectedError bool
	}{
		{0, 0, 0, true},
		{3, 0, 0, true},
		{3, 2, 0, true},
		{3, 2, 1, false},
	}

	for _, test := range tests {
		_, err := NewFFN(test.InputNeurons, test.OutputNeurons, test.HiddenNeurons, nil)
		if (err == nil && test.ExpectedError) || (err != nil && !test.ExpectedError) {
			t.Errorf("Expected error: %t, received: %v", test.ExpectedError, err)
		}
	}
}

func TestTrain(t *testing.T) {
	tests := []struct {
		Inputs        [][]float64
		Outputs       [][]float64
		Speed         float64
		Moment        float64
		Rate          float64
		Epoch         int
		ExpectedError bool
	}{
		{[][]float64{[]float64{0}}, [][]float64{}, 0, 0, 0, 0, true},
		{[][]float64{[]float64{0}}, [][]float64{[]float64{1}}, 0, 0, 0, 0, true},
		{[][]float64{[]float64{0}}, [][]float64{[]float64{1}}, 0.2, 0, 0, 0, true},
		{[][]float64{[]float64{0}}, [][]float64{[]float64{1}}, 0.2, -0.5, 0, 0, true},
		{[][]float64{[]float64{0}}, [][]float64{[]float64{1}}, 0.2, 0.05, -0.5, 0, true},
		{[][]float64{[]float64{0}}, [][]float64{[]float64{1}}, 0.2, 0.05, 0.01, 0, true},
		{[][]float64{[]float64{1}}, [][]float64{[]float64{1}}, 0.2, 0, 100, 100000, false},
	}

	n, err := NewFFN(1, 1, 1, nil)
	if err != nil {
		t.Error(err)
	}

	for _, test := range tests {
		_, _, err := n.Train(test.Inputs, test.Outputs, test.Speed, test.Moment, test.Rate, test.Epoch, nil)
		if (err == nil && test.ExpectedError) || (err != nil && !test.ExpectedError) {
			t.Errorf("Expected error: %t, received: %v", test.ExpectedError, err)
		}
	}
}

func TestFromFile(t *testing.T) {
	tests := []struct {
		Filename      string
		ExpectedError bool
	}{
		{"", true},
		{"notfound.txt", true},
		{"etc/dump_test.json", false},
	}

	n, err := NewFFN(1, 1, 1, nil)
	if err != nil {
		t.Error(err)
	}

	for _, test := range tests {
		err = n.FromFile(test.Filename)
		if (err == nil && test.ExpectedError) || (err != nil && !test.ExpectedError) {
			t.Errorf("Expected error: %t, received: %v", test.ExpectedError, err)
		}
	}
}
