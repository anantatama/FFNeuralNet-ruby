require 'csv'

n_hidden = 3
n_output = 2

def read_dataset
  CSV.parse(File.read('iris.csv'), headers: true)
end

def init_weight(n, m)
  w_hidden = []
  w_output = []

  (1..n).each do |i|
    w_hidden << Array.new(4) { rand(-1.0..1.0) }
  end

  (1..m).each do |j|
    w_output << Array.new(3) { rand(-1.0..1.0) }
  end

  return w_hidden, w_output
end

def sigmoid(x)
  1 / (1 + Math.exp(1)**(-x))
end

def error_function(species, ouput_neuron_1, ouput_neuron_2)
  case species
  when "setosa"
    (0-ouput_neuron_1).abs + (0-ouput_neuron_2).abs
  when "versicolor"
    (0-ouput_neuron_1).abs + (1-ouput_neuron_2).abs
  when "virginica"
    (1-ouput_neuron_1).abs + (1-ouput_neuron_2).abs
  end
end

def train(dataset, hidden, output)
  error = []
  dataset.each do |data|
    hidden_layer = []
    hidden_layer << data[0].to_f * hidden[0][0] + data[1].to_f * hidden[0][1] + data[2].to_f * hidden[0][2] + data[3].to_f * hidden[0][3]
    hidden_layer << data[0].to_f * hidden[1][0] + data[1].to_f * hidden[1][1] + data[2].to_f * hidden[1][2] + data[3].to_f * hidden[1][3]
    hidden_layer << data[0].to_f * hidden[2][0] + data[1].to_f * hidden[2][1] + data[2].to_f * hidden[2][2] + data[3].to_f * hidden[2][3]

    output_layer = []
    output_layer << hidden_layer[0] * output[0][0] + hidden_layer[0] * output[0][1] + hidden_layer[0] * output[0][2]
    output_layer << hidden_layer[1] * output[1][0] + hidden_layer[1] * output[1][1] + hidden_layer[1] * output[1][2]

    act_1 = sigmoid(output_layer[0])
    act_2 = sigmoid(output_layer[1])
    error << error_function(data[4], act_1, act_2)
  end
  total_error = error.sum / dataset.length
end

iris = read_dataset
w_hidden, w_output = init_weight(n_hidden, n_output)
result = train(iris, w_hidden, w_output)
puts result