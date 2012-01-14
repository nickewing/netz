(ns netz.core
  (:gen-class)
  (:use incanter.core))

(defn report-callback
  [network epoch mse complete]
  (if complete
    (println "Finished Epoch" epoch "MSE" mse)
    (println "Epoch" epoch "MSE" mse))
  true)

(def default-options
  {:max-epochs 20000
   :desired-error 0.005
   :learning-rate 0.25
   :learning-momentum 0
   :callback report-callback
   :callback-resolution 100
   :regularization-constant 0})

(defn- option
  [network option-name]
  (or (option-name (:options network))
      (option-name default-options)))

(defn- matrix-mult
  "Multiply two matrices and ensure the result is also a matrix."
  [a b]
  (let [result (mmult a b)]
    (if (matrix? result)
      result
      (matrix [result]))))

(defn- round-output
  "Round outputs to nearest integer."
  [output]
  (vec (map #(Math/round ^Double %) output)))

(load "forward_prop")
(load "back_prop")
(load "gradient_descent")

(defprotocol NeuralNetwork
  (run [network inputs])
  (run-binary [network inputs])
  (train-on-examples [network inputs outputs]))

(defrecord MultiLayerPerceptron [weights options]
  NeuralNetwork
  (run [network inputs]
    (let [weights (:weights network)
          input-activations (matrix inputs)]
      (forward-propagate weights input-activations)))
  (run-binary [network inputs]
    (round-output (run network inputs)))
  (train-on-examples [network inputs outputs]
    (let [num-inputs (length (first inputs))
          num-outputs (length (first outputs))
          hidden-neurons (:hidden-neurons (:options network))
          layer-sizes (conj (vec (cons num-inputs hidden-neurons)) num-outputs)
          network (assoc network :weights (random-weight-matrices layer-sizes))
          examples (map #(vector %1 %2) inputs outputs)]
      (gradient-descent-backpropagation network examples))))

(defn train [inputs outputs & [options]]
  (let [network (MultiLayerPerceptron. nil options)]
    (train-on-examples network inputs outputs)))

(defn new-network [weights & [options]]
  (MultiLayerPerceptron. weights (or options {})))
