package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"math"
)

type Activation func(x float64) float64

type Layer interface {
	predict_proba([]float64) []float64
	fit(train_x []float64, train_y []float64, eps float64, verbose_level int)
}

type Neuron struct {
	input_size int
	weights    []float64
	bias       float64
	activation Activation
}

func (neuron *Neuron) getResult(data []float64) float64 {
	result := -neuron.bias
	for i := 0; i < neuron.input_size; i++ {
		result += data[i] * neuron.weights[i]
	}
	return neuron.activation(result)
}

func (neuron *Neuron) fit(train_x []float64, y float64, eps float64) {
	result := neuron.getResult(train_x)
	delta_bias := eps * (y - result)
	for i := 0; i < neuron.input_size; i++ {
		delta_weight := delta_bias * train_x[i]
		neuron.weights[i] = neuron.weights[i] + delta_weight
	}
	neuron.bias = neuron.bias - delta_bias
}

type InputLayer struct {
	size    int
	neurons []Neuron
}

func NewInputLayer(size_input int, size_output int, activate Activation) (*InputLayer){
	neurons := make([]Neuron, size_output)
	for i, neuron := range neurons{
		neuron.activation = activate
		neuron.input_size = size_input
		neuron.weights = make([]float64, size_input)
		neurons[i] = neuron
	}
	return &InputLayer{size_output, neurons}
}

func (layer *InputLayer) predict_proba(test_x []float64) []float64 {
	result := make([]float64, layer.size)
	for i, neuron := range layer.neurons {
		result[i] = neuron.getResult(test_x)
	}
	return result
}

func (layer *InputLayer) predict(test_x []float64) []float64 {
	probs := layer.predict_proba(test_x)
	max_ind := 0
	max_val := 0.
	for i, val := range probs {
		if val > max_val{
			max_val = val
			max_ind = i
		}
	}
	result := make([]float64, len(probs))
	result[max_ind] = 1
	return result
}

func getResultFromPredict(predict []float64) (result int){
	for ind, val := range predict {
		if val == 1 {
			result = ind
		}
	}
	return
}

func (layer *InputLayer) fit(train_x []float64, train_y []float64, eps float64, verbose_level int) {
	for i, neuron := range layer.neurons {
		neuron.fit(train_x, train_y[i], eps)
	}
	return
}

func compareArrays(arr1 []float64, arr2 []float64) bool{
	result := true
	for i, val := range arr1{
		if val != arr2[i]{
			result = false
		}
	}
	return result
}

func (layer *InputLayer) learn(train_data [][]float64, train_result [][]float64, epochs int, eps float64, verbose_level int) {
	for i := 0; i < epochs; i++ {
		func() {
			if verbose_level > 0 {
				fmt.Printf("Эпоха обучения %d\n", i+1)
				defer func(){
					count_valid := 0
					for i, train_x := range train_data {
						if compareArrays(layer.predict(train_x), train_result[i]){
							count_valid++
					    	}
					}
					fmt.Printf("Точность: %.2f%%\n", 100*float64(count_valid)/float64(len(train_data)))
				}()
			}
			count := 0
			for i, train_x := range train_data {
				count ++
				layer.fit(train_x, train_result[i], eps, verbose_level)
			}
			fmt.Println(count)
		}()
	}
}

const MAX_DARKNESS float64 = 255

func getTrainData(path string, header bool) (train_x [][]float64, train_y [][]float64) {
	f, _ := os.Open(path)
	r := csv.NewReader(bufio.NewReader(f))
	if header {
		r.Read()
	}
	for {
		record, err := r.Read()
		// Stop at EOF.
		if err == io.EOF {
			break
		}
		value, err := strconv.Atoi(record[0])
		value_arr := make([]float64, 10)
		value_arr[value] = 1.
		if err != nil {
			fmt.Println("Not valid csv file")
			os.Exit(1)
		}
		image := record[1:]
		image_normalized := make([]float64, len(image))
		for ind, pix := range image {
			pix, err := strconv.Atoi(pix)
			if err != nil {
				fmt.Println("Not valid csv file")
				os.Exit(1)
			}
			image_normalized[ind] = float64(pix) / MAX_DARKNESS
		}
		train_y = append(train_y, value_arr)
		train_x = append(train_x, image_normalized)
	}
	return
}

func getTestData(path string, header bool) (train_x [][]float64) {
	f, _ := os.Open(path)
	r := csv.NewReader(bufio.NewReader(f))
	if header {
		r.Read()
	}
	for {
		record, err := r.Read()
		// Stop at EOF.
		if err == io.EOF {
			break
		}
		image := record[0:]
		image_normalized := make([]float64, len(image))
		for ind, pix := range image {
			pix, err := strconv.Atoi(pix)
			if err != nil {
				fmt.Println("Not valid csv file")
				os.Exit(1)
			}
			image_normalized[ind] = float64(pix) / MAX_DARKNESS
		}
		train_x = append(train_x, image_normalized)
	}
	return
}

func Stepper (x float64) float64{
	if x>0{
		return 1.
	} else {
		return 0.
	}
}

func Sigmoid (x float64) float64{
	return 1.0 / (1.0 + math.Exp(-x))
}

func main() {
	train_x, train_y := getTrainData("data/train.csv", true)
	layer := NewInputLayer(len(train_x[0]), len(train_y[0]), Sigmoid)
	layer.learn(train_x, train_y, 50, 0.01, 1)
	test_x := getTestData("data/test.csv", true)


	file, _ := os.Create("result.csv")
	defer file.Close()

	writer := csv.NewWriter(file)
	writer.Write([]string{"ImageId","Label"})
	for ind, value := range test_x {
		result := layer.predict(value)
		writer.Write([]string{strconv.Itoa(ind+1), strconv.Itoa(getResultFromPredict(result))})
	}

	defer writer.Flush()
}

type Network struct {

}


