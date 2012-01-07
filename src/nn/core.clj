(ns nn.core
  (:gen-class)
  (:use (incanter core)))

(set! *warn-on-reflection* true)

(defn- sigmoid
  "Sigmoid function: 1/(1+exp(-z))."
  [z]
  (div 1 (plus 1 (exp (minus z)))))

(defn- bind-bias
  "Prepend the bias unit to a layer activation vector."
  [v]
  (bind-rows [1] v))

(defn- propagate-layer
  "Calculate activations for layer l+1 given weight matrix between layer l
  and l+1 and layer l activations."
  [weight-matrix activations]
  (sigmoid (mmult weight-matrix (bind-bias activations))))

(defn run
  "Run forward propagation using the given network and inputs."
  [network inputs]
  (let [weights (:weights network)
        activations (matrix inputs)]
    (reduce #(propagate-layer %2 %1) activations weights)))

(defn- round-output [output]
  (if (matrix? output)
    (map #(Math/round ^Double %) output)
    (Math/round ^Double output)))

(defn run-binary
  "Run forward propagation on given network and inputs and then round
  output value(s)."
  [network inputs]
  (round-output (run network inputs)))

(defn -main [& args]
  (println (run-binary [0 0])))
