(ns netz.core
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
   :callback report-callback
   :callback-resolution 100
   :regularization-constant 0.0
   :calc-batch-error-in-parallel true
   :training-algorithm :rprop
   :bprop {:learning-rate 0.25
           :learning-momentum 0.0}
   :rprop {:init-update 0.1
           :update-min 1e-6
           :update-max 50.0
           :increase-factor 1.2
           :decrease-factor 0.5}
   :weight-initialization {:method :nguyen-widrow
                           :range 0.5}})

(defn- get-option
  [network & option-path]
  (or (reduce #(%2 %1) (:options network) option-path)
      (reduce #(%2 %1) default-options option-path)))

(load "util")
(load "forward_prop")
(load "weight_init")
(load "back_prop")
(load "gradient_descent")

(defprotocol NeuralNetwork
  (run [network inputs])
  (run-binary [network inputs])
  (train-on-examples [network examples]))

(defrecord MultiLayerPerceptron [weights options]
  NeuralNetwork
  (run [network inputs]
    (let [weights (:weights network)
          input-activations (matrix inputs)]
      (forward-propagate weights input-activations)))
  (run-binary [network inputs]
    (round-output (run network inputs)))
  (train-on-examples [network examples]
    (let [[first-input first-output] (first examples)
          num-inputs (length first-input)
          num-outputs (length first-output)
          examples (map #(list (matrix (first %)) (matrix (second %))) examples)
          hidden-neurons (or (:hidden-neurons (:options network))
                             (vec num-inputs))
          layer-sizes (conj (vec (cons num-inputs hidden-neurons)) num-outputs)
          init-fn (case (get-option network :weight-initialization :method)
                    :random random-initial-weights
                    :nguyen-widrow nguyen-widrow-initial-weights)
          initial-weights (init-fn layer-sizes
                                   (get-option network :weight-initialization :range))
          network (assoc network :weights initial-weights)
          training-fn (case (get-option network :training-algorithm)
                        :bprop gradient-descent-bprop
                        :rprop gradient-descent-rprop)]
      (training-fn network examples))))

(defn train [examples & [options]]
  (let [network (MultiLayerPerceptron. nil options)]
    (train-on-examples network examples)))

(defn new-network [weights & [options]]
  (MultiLayerPerceptron. weights (or options {})))
