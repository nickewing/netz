(ns nn.core
  (:gen-class)
  (:use (incanter core)))

(set! *warn-on-reflection* true)

(def init-epsilon 0.2)

(def max-epochs 20000)
(def desired-error 1e-2)
(def learning-rate 0.25)
(def learning-momentum 0.9)
(def epochs-per-report 100)

(defn- sigmoid
  "Sigmoid function: 1/(1+exp(-z))."
  [z]
  (div 1 (plus 1 (exp (minus z)))))

(defn- bind-bias
  "Prepend the bias unit to a layer activation vector."
  [v]
  (bind-rows [1] v))

(defn- mmult-matrix
  "Multiply two matrices and ensure the result is also a matrix."
  [a b]
  (let [result (mmult a b)]
    (if (matrix? result)
      result
      (matrix [result]))))

(defn- forward-propagate-layer
  "Calculate activations for layer l+1 given weight matrix between layer l
  and l+1 and layer l activations."
  [weight-matrix activations]
  (sigmoid (mmult-matrix weight-matrix activations)))

(defn- forward-propagate
  "Propagate activation values through the network and return all activation
  values for all nodes."
  [weights input-activations]
  (let [input-activations (bind-bias input-activations)]
    ((fn [all-weights next-activations all-activations]
       (let [weights (first all-weights)
             all-weights (rest all-weights)
             last-iter (empty? all-weights)
             next-activations (forward-propagate-layer weights next-activations)
             next-activations (if last-iter
                                next-activations
                                (bind-bias next-activations))
             all-activations (conj all-activations next-activations)]
         (if last-iter
           all-activations
           (recur all-weights next-activations all-activations))))
     weights input-activations [input-activations])))

(defn- output-from-activations
  "Return output layer activation values."
  [activations]
  (map identity (last activations)))

(defn run
  "Run forward propagation using the given network and inputs."
  [network inputs]
  (let [weights (:weights network)
        input-activations (matrix inputs)]
    (output-from-activations (forward-propagate weights input-activations))))

(defn- round-output
  "Round outputs to nearest integer."
  [output]
  (vec (map #(Math/round ^Double %) output)))

(defn run-binary
  "Run forward propagation on given network and inputs and then round
  output value(s)."
  [network inputs]
  (round-output (run network inputs)))


(defn- random-list
  "Create a list of random doubles between -init-epsilon and +init-epsilon."
  [len]
  (for [x (range 0 len)]
    (- (rand (* 2 init-epsilon)) init-epsilon)))

(defn- random-weight-matrices
  "Generate random initial weight matrices for given layer-sizes."
  [layer-sizes]
  (for [i (range 0 (dec (length layer-sizes)))]
    (let [cols (inc (get layer-sizes i))
          rows (get layer-sizes (inc i))]
      (matrix (random-list (* rows cols)) cols))))

(defn- back-propagate-layer-deltas
  "Back propagate last-deltas (from layer l-1) and return layer l deltas."
  [last-deltas weight-matrix layer-activations]
  (mult (mmult-matrix (trans weight-matrix) last-deltas)
        (mult layer-activations (minus 1 layer-activations))))

(defn- calc-hidden-deltas
  "Calculate hidden deltas for back propagation.  Returns all deltas including
  output-deltas."
  [weights activations output-deltas]
  (let [hidden-weights (reverse (rest weights))
        hidden-activations (rest (reverse (rest activations)))]
    ((fn [last-deltas all-weights all-activations all-deltas]
       (if (empty? all-weights)
         all-deltas
         (let [weights (first all-weights)
               all-weights (rest all-weights)
               activations (first all-activations)
               all-activations (rest all-activations)
               last-deltas (back-propagate-layer-deltas last-deltas weights activations)
               all-deltas (cons (rest last-deltas) all-deltas)]
           (recur last-deltas all-weights all-activations all-deltas))))
     output-deltas hidden-weights hidden-activations (list output-deltas))))

(defn- add-delta-sum
  "Add new delta sum to accumulator."
  [delta-sums deltas activations]
  (map #(plus %1 (mmult %2 (trans %3)))
       delta-sums deltas activations))

(defn- new-weight-accumulator
  "Create accumulator matrix list of the same structure as the given weight list
  with all zero values."
  [weights]
  (map (fn [weight]
         (let [[rows cols] (dim weight)]
           (matrix 0 rows cols)))
       weights))

(defn- back-propagate-example
  "Run back propagation on one sample and return delta sums."
  [delta-sums weights input expected-output]
  (let [activations (forward-propagate weights (matrix input))
        output (last activations)
        output-deltas (minus output expected-output)
        all-deltas (calc-hidden-deltas weights activations output-deltas)
        delta-sums (add-delta-sum delta-sums all-deltas activations)]
    delta-sums))


; TODO: handle regularization
(defn- back-propagate-all-examples
  "Run back propagation on all examples and return gradients."
  [weights examples]
  (let [num-examples (length examples)
        delta-sums (reduce
                     #(back-propagate-example %1 weights (first %2) (second %2))
                     (new-weight-accumulator weights) examples)
        gradients (map #(div % num-examples) delta-sums)]
    gradients))

(defn- calc-mse
  "Calculate mean squared error of predictions"
  [weights examples]
  (let [sum (reduce (fn [total example]
                      (let [[input expected-output] example
                            output (output-from-activations
                                               (forward-propagate weights (matrix input)))]
                        (+ total (sum (pow (minus output expected-output) 2)))))
                    0 examples)]
    (/ sum (length examples))))

(defn- gradient-descent
  "Preform gradient descent to adjust network weights"
  ([weights examples last-changes epoch]
   (let [gradients (back-propagate-all-examples weights examples)
         changes (map (fn [weights gradients last-change]
                        (plus (mult learning-rate gradients)
                              (mult last-change learning-momentum)))
                      weights gradients last-changes)
         new-weights (map #(minus %1 %2) weights changes)
         mse (calc-mse new-weights examples)]
     (if (= (mod epoch epochs-per-report) 0)
       (println "Epoch " epoch "MSE" mse))
     (if (or (> epoch max-epochs) (< mse desired-error))
       (do
         (println "Total epochs:" epoch "Final MSE:" mse)
         new-weights)
       (recur new-weights examples changes (inc epoch)))))
  ([weights examples]
   (gradient-descent weights examples (new-weight-accumulator weights) 0)))

(defn train
  "Train network on training examples."
  [inputs outputs hidden-neurons]
  (let [num-inputs (length (first inputs))
        num-outputs (length (first outputs))
        layer-sizes (conj (vec (cons num-inputs hidden-neurons)) num-outputs)
        weights (random-weight-matrices layer-sizes)
        examples (map #(vector %1 %2) inputs outputs)]
    {:weights (gradient-descent weights examples)}))


